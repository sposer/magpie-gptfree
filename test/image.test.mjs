import { strict as assert } from "node:assert";
import { test } from "node:test";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile, rm, writeFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { setTimeout as delay } from "node:timers/promises";
import {
  CODEX_BASE, discoverPoolModels, poolFetch, probeAccountImageQuota, refreshPoolAccount,
} from "../lib/codex-pool.mjs";
import {
  IMAGE_MODEL_ID,
  ImageHttpError,
  buildImageModelCatalog,
  checkWebImageQuota,
  executeWebImageGeneration,
  isImageModelId,
  parseImageRequest,
  runWebImageWorker,
  supportsGifOutput,
} from "../lib/chatgpt-web-image.mjs";
import {
  createInMemoryStatsStore, isAccountRateLimited, readPoolStats, recordQuotaCheck,
  sanitizeStatsMessage, summarizePoolStats,
} from "../lib/pool-stats.mjs";
import { accountSlotTag, buildSlotAuthBody } from "../lib/pool-connection.mjs";
import { GPTFreePoolPlugin } from "../index.mjs";

const seed = "GEZDGNBVGY3TQOJQGEZDGNBVGY3TQOJQ";
function jwt(delta, email = "a@example.com") {
  return [
    Buffer.from('{"alg":"RS256","typ":"JWT"}').toString("base64url"),
    Buffer.from(JSON.stringify({
      exp: Math.floor(Date.now() / 1000) + delta, email,
      "https://api.openai.com/auth": { chatgpt_account_id: "acct123" },
    })).toString("base64url"),
    "signature",
  ].join(".");
}

function findPython() {
  for (const cmd of process.platform === "win32" ? ["python", "python3"] : ["python3", "python"]) {
    const res = spawnSync(cmd, ["-c", "import sys;print(sys.version_info.major)"], { encoding: "utf8" });
    if (res.status === 0 && String(res.stdout).trim() === "3") return cmd;
  }
  return null;
}

test("image model catalog exposes Magpie host.js @ai-sdk/openai-compatible schema and capabilities", async () => {
  const cat = buildImageModelCatalog(CODEX_BASE);
  const img = cat[IMAGE_MODEL_ID];
  assert.equal(img.id, "gpt-image-2");
  assert.equal(img.api.id, "gpt-image-2");
  assert.equal(img.api.url, CODEX_BASE);
  assert.equal(img.api.npm, "@ai-sdk/openai-compatible");
  assert.equal(img.capabilities.input.text, true);
  assert.equal(img.capabilities.input.image, true);
  assert.equal(img.capabilities.output.text, false);
  assert.equal(img.capabilities.output.image, true);
  assert.equal(img.attachment, true);

  // The 2.6 family must be exposed too, and only 2.6 supports animated GIF.
  const flags = ["gpt-image-2.6", "gpt-image-2.6-flare", "gpt-image-2.6-sunburst"];
  for (const id of flags) {
    const model = cat[id];
    assert.equal(model.id, id);
    assert.equal(model.api.npm, "@ai-sdk/openai-compatible");
    assert.equal(model.capabilities.output.image, true);
    assert.equal(model.capabilities.animated, true);
  }
  assert.equal(cat["gpt-image-2"].capabilities.animated, undefined);
  assert.equal(isImageModelId("plus-gpt-image-2.6"), true);

  // GIF output is accepted for 2.6 and rejected for the older model.
  const gif2 = await parseImageRequest("/backend-api/codex/images/generations", JSON.stringify({
    model: "gpt-image-2.6",
    prompt: "spinning coin",
    output_format: "gif",
    frame_count: 12,
    frame_duration_ms: 100,
  }));
  assert.equal(gif2.model, "gpt-image-2.6");
  assert.equal(gif2.output_format, "gif");
  assert.equal(gif2.frame_count, 12);
  assert.equal(gif2.frame_duration_ms, 100);
  await assert.rejects(
    parseImageRequest("/backend-api/codex/images/generations", JSON.stringify({
      model: "gpt-image-2",
      prompt: "spinning coin",
      output_format: "gif",
    })),
    (err) => err.status === 400 && /2\.6/.test(err.message),
  );
  await assert.rejects(
    parseImageRequest("/backend-api/codex/images/generations",
      JSON.stringify({ model: "gpt-image-9", prompt: "x" })),
    (err) => err.status === 400 && err.code === "model_not_found",
  );

  // Verify integrated discoverPoolModels keeps Codex text models on @ai-sdk/openai
  const models = await discoverPoolModels({
    read: async () => [{ email: "free@example.com", password: "p", totp: seed, accessToken: jwt(3600, "free@example.com") }],
    fetcher: async () => new Response(JSON.stringify({
      models: [{ slug: "gpt-5.6-luna", display_name: "GPT-5.6-Luna", visibility: "list" }],
    }), { status: 200, headers: { "content-type": "application/json" } }),
  });
  assert.equal(models["gpt-5.6-luna"].api.npm, "@ai-sdk/openai");
  assert.equal(models["gpt-image-2"].api.npm, "@ai-sdk/openai-compatible");
});

test("TestModels chat/completions probe returns 400 pointing to Images API or Responses API", async () => {
  const imgProbe = await poolFetch(CODEX_BASE + "/chat/completions", {
    method: "POST",
    body: JSON.stringify({ model: "gpt-image-2", messages: [{ role: "user", content: "hi" }] }),
  });
  assert.equal(imgProbe.status, 400);
  const imgJson = await imgProbe.json();
  assert.match(imgJson.error.message, /\/v1\/images\/generations/);
  assert.match(imgJson.error.message, /\/v1\/images\/edits/);

  const codexProbe = await poolFetch(CODEX_BASE + "/chat/completions", {
    method: "POST",
    body: JSON.stringify({ model: "gpt-5.6-luna", messages: [{ role: "user", content: "hi" }] }),
  });
  assert.equal(codexProbe.status, 400);
  const codexJson = await codexProbe.json();
  assert.match(codexJson.error.message, /\/v1\/responses/);
});

test("parseImageRequest parses both JSON generations and multipart/form-data edits with image[], mask, and markup", async () => {
  const gen = await parseImageRequest("/backend-api/codex/images/generations", JSON.stringify({
    model: "gpt-image-2",
    prompt: "a corgi in space",
    n: 2,
    size: "1024x1024",
    quality: "high",
    output_format: "webp",
    background: "transparent",
  }));
  assert.equal(gen.model, "gpt-image-2");
  assert.equal(gen.n, 2);
  assert.equal(gen.output_format, "webp");
  assert.equal(gen.background, "transparent");
  assert.deepEqual(gen.images, []);

  // Build real multipart/form-data payload using FormData + Response
  const fd = new FormData();
  fd.append("model", "gpt-image-2");
  fd.append("prompt", "add sunglasses");
  fd.append("n", "1");
  fd.append("image[]", new File([Buffer.from("png-bytes-1")], "photo.png", { type: "image/png" }));
  fd.append("mask", new File([Buffer.from("mask-bytes-1")], "mask.png", { type: "image/png" }));
  fd.append("markup", new File([Buffer.from("markup-bytes-1")], "markup.png", { type: "image/png" }));
  const multipartReq = new Request("https://chatgpt.com/backend-api/codex/images/edits", {
    method: "POST",
    body: fd,
  });
  const rawBuf = Buffer.from(await multipartReq.arrayBuffer());
  const contentType = multipartReq.headers.get("content-type");

  const edit = await parseImageRequest("/backend-api/codex/images/edits", rawBuf, { "content-type": contentType });
  assert.equal(edit.model, "gpt-image-2");
  assert.equal(edit.prompt, "add sunglasses");
  assert.equal(edit.images.length, 2);
  assert.equal(edit.images[0].filename, "photo.png");
  assert.equal(edit.images[1].filename, "markup.png");
  assert.equal(edit.masks.length, 1);
  assert.equal(edit.masks[0].filename, "mask.png");

  // Remote image URL in JSON edits is rejected with 400
  await assert.rejects(
    parseImageRequest("/backend-api/codex/images/edits", JSON.stringify({
      prompt: "edit",
      image: "https://attacker.invalid/secret.png",
    })),
    (err) => err.status === 400,
  );
});

test("native multi-image requests renew only the selected account and keep its proxy", async () => {
  const accA = { email: "alpha@example.com", password: "p1", totp: seed, accessToken: jwt(3600, "alpha@example.com") };
  const accB = { email: "beta@example.com", password: "p2", totp: seed, accessToken: jwt(3600, "beta@example.com") };
  const renewedB = jwt(3600, "beta@example.com") + "2";
  const slotB = accountSlotTag(accB.email);

  const calls = [];
  let reauthCalls = 0;

  const res = await poolFetch(CODEX_BASE + "/images/generations", {
    method: "POST",
    body: JSON.stringify({ model: "gpt-image-2", prompt: "cute orange cat", n: 2, size: "1024x1024" }),
  }, {
    read: async () => [accA, accB],
    preferredSlot: slotB,
    proxy: "http://slot-b-proxy:7890",
    reauthenticate: async (account, options) => {
      assert.equal(account.email, accB.email);
      assert.equal(options.proxy, "http://slot-b-proxy:7890");
      assert(options.signal instanceof AbortSignal);
      reauthCalls++;
      return { ...account, accessToken: renewedB };
    },
    runWorker: async (payload) => {
      assert.equal(payload.proxy, "http://slot-b-proxy:7890");
      calls.push({ token: payload.access_token, prompt: payload.prompt });
      if (payload.access_token === accB.accessToken) {
        return { ok: false, http_status: 401, error_code: "unauthorized", error: "token expired" };
      }
      return {
        ok: true,
        created: 1731150000,
        data: [{ b64_json: Buffer.from("fake-png").toString("base64"), media_type: "image/png", revised_prompt: payload.prompt }],
      };
    },
  });

  assert.equal(res.status, 200);
  const body = await res.json();
  assert.equal(body.created, 1731150000);
  assert.equal(body.data.length, 2);
  assert.equal(body.data[0].media_type, "image/png");
  assert(body.usage.input_tokens > 0);
  assert(body.usage.output_tokens >= 2048);
  assert.equal(reauthCalls, 1);
  assert.equal(calls[0].token, accB.accessToken);
  assert(calls.every(({ token }) => token === accB.accessToken || token === renewedB));
});

test("content policy refusal (400) stops immediately without burning remaining pooled accounts", async () => {
  const acc1 = { email: "a@example.com", password: "p", totp: seed, accessToken: jwt(3600, "a@example.com") };
  const acc2 = { email: "b@example.com", password: "p", totp: seed, accessToken: jwt(3600, "b@example.com") };
  let workerCalls = 0;

  const res = await poolFetch(CODEX_BASE + "/images/generations", {
    method: "POST",
    body: JSON.stringify({ model: "gpt-image-2", prompt: "blocked prompt" }),
  }, {
    read: async () => [acc1, acc2],
    runWorker: async () => {
      workerCalls++;
      return {
        ok: false,
        http_status: 400,
        error_code: "content_policy_violation",
        error: "Rejected by safety system",
      };
    },
  });

  assert.equal(res.status, 400);
  assert.equal(workerCalls, 1);
  const json = await res.json();
  assert.equal(json.error.code, "content_policy_violation");
});

test("GPTFreePoolPlugin loader wires poolSlot into poolFetch for /images/edits", async () => {
  const plug = await GPTFreePoolPlugin();
  const auth = buildSlotAuthBody("slot@example.com");
  const loaderOpts = await plug.auth.loader(async () => auth);
  assert.equal(loaderOpts.baseURL, CODEX_BASE);
  // Calling loader fetch with an invalid URL still goes through poolFetch validation
  await assert.rejects(loaderOpts.fetch("https://attacker.invalid/images/generations"), /only forwards/);
});

const pythonCmd = findPython();

test("runWebImageWorker supports real Python IPC and reaps timeout, abort and oversized-output processes", {
  skip: !pythonCmd,
}, async (t) => {
  const tmp = await mkdtemp(join(tmpdir(), "gptfree-py-worker-"));
  try {
    const badPromptRes = await runWebImageWorker(
      { action: "generate_image", access_token: "tok", prompt: "   " },
      { python: async () => pythonCmd },
    );
    assert.equal(badPromptRes.ok, false);
    assert.equal(badPromptRes.http_status, 400);

    const sleepScript = join(tmp, "slow_worker.py");
    await writeFile(sleepScript, [
      "import json, os, signal, sys, time",
      "from pathlib import Path",
      "payload = json.load(sys.stdin)",
      "if os.name != 'nt': signal.signal(signal.SIGTERM, signal.SIG_IGN)",
      "Path(payload['pid_file']).write_text(str(os.getpid()), encoding='utf-8')",
      "if payload['mode'] == 'output_too_large':",
      "    for _ in range(65): sys.stdout.buffer.write(b'x' * (1024 * 1024))",
      "    sys.stdout.buffer.flush()",
      "time.sleep(30)",
    ].join("\n"), "utf8");

    async function readWorkerPid(path) {
      const deadline = performance.now() + 5_000;
      while (performance.now() < deadline) {
        try {
          const pid = Number(await readFile(path, "utf8"));
          if (Number.isInteger(pid) && pid > 0) return pid;
        } catch (err) {
          if (err.code !== "ENOENT") throw err;
        }
        await delay(10);
      }
      assert.fail("Python worker did not write its PID");
    }

    for (const mode of ["worker_timeout", "abort", "output_too_large"]) {
      await t.test(mode, async () => {
        const pidFile = join(tmp, `${mode}.pid`);
        const ac = new AbortController();
        const pending = runWebImageWorker(
          { prompt: "一只可爱的橘猫", pid_file: pidFile, mode },
          {
            python: async () => pythonCmd,
            workerPath: sleepScript,
            signal: ac.signal,
            timeout: mode === "worker_timeout" ? 1_000 : 10_000,
          },
        );
        const outcome = pending.then((value) => ({ value }), (error) => ({ error }));
        try {
          const pid = await readWorkerPid(pidFile);
          if (mode === "abort") ac.abort();
          const { value, error } = await outcome;
          if (mode === "abort") {
            assert.equal(error?.name, "AbortError");
          } else {
            assert.ifError(error);
            assert.equal(value.ok, false);
            assert.equal(value.http_status, mode === "worker_timeout" ? 504 : 502);
            assert.equal(value.error_code, mode);
          }
          assert.throws(() => process.kill(pid, 0), { code: "ESRCH" });
        } finally {
          ac.abort();
          await outcome;
        }
      });
    }
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

test("runWebImageWorker bounds setup waits without spawning cancelled callers or cancelling shared setup", {
  skip: !pythonCmd,
}, async (t) => {
  const tmp = await mkdtemp(join(tmpdir(), "gptfree-py-setup-"));
  try {
    const script = join(tmp, "marker_worker.py");
    await writeFile(script, [
      "import json, sys",
      "from pathlib import Path",
      "payload = json.load(sys.stdin)",
      "Path(payload['marker']).write_text('spawned', encoding='utf-8')",
      "print(json.dumps({'ok': True, 'data': [{'b64_json': 'cG5n'}]}))",
    ].join("\n"), "utf8");

    for (const mode of ["abort", "worker_timeout"]) {
      await t.test(mode, async () => {
        let finishSetup;
        const sharedSetup = new Promise((resolve) => { finishSetup = resolve; });
        const ac = new AbortController();
        const survivorAbort = new AbortController();
        const cancelledMarker = join(tmp, `${mode}-cancelled`);
        const survivorMarker = join(tmp, `${mode}-survivor`);
        const cancelled = runWebImageWorker({ marker: cancelledMarker }, {
          python: () => sharedSetup, workerPath: script, signal: ac.signal,
          timeout: mode === "worker_timeout" ? 30 : 5_000,
        });
        const outcome = cancelled.then((value) => ({ value }), (error) => ({ error }));
        const survivor = runWebImageWorker({ marker: survivorMarker }, {
          python: () => sharedSetup, workerPath: script, signal: survivorAbort.signal, timeout: 5_000,
        });
        const survivorOutcome = survivor.then((value) => ({ value }), (error) => ({ error }));
        try {
          if (mode === "abort") ac.abort();
          const { value, error } = await outcome;
          if (mode === "abort") {
            assert.equal(error?.name, "AbortError");
          } else {
            assert.ifError(error);
            assert.equal(value.http_status, 504);
            assert.equal(value.error_code, "worker_timeout");
          }
          finishSetup(pythonCmd);
          const survived = await survivorOutcome;
          assert.ifError(survived.error);
          assert.equal(survived.value.ok, true);
          await assert.rejects(readFile(cancelledMarker), { code: "ENOENT" });
          await readFile(survivorMarker);
        } finally {
          ac.abort();
          survivorAbort.abort();
          finishSetup(pythonCmd);
          await Promise.all([outcome, survivorOutcome]);
        }
      });
    }
  } finally {
    await rm(tmp, { recursive: true, force: true });
  }
});

test("executeWebImageGeneration shares a hard budget with the second account attempt", async () => {
  const accounts = [1, 2, 3].map((n) => ({
    email: `u${n}@example.com`,
    accessToken: jwt(3600, `u${n}@example.com`),
  }));
  const timeouts = [];
  const reports = [];
  let deadlineSignal;
  await assert.rejects(
    executeWebImageGeneration({
      accounts,
      request: { model: IMAGE_MODEL_ID, prompt: "cat", n: 1 },
      maxTotalMs: 150,
      onAccountResult: (evt) => reports.push(evt),
      runWorker: async (_payload, { signal, timeout }) => {
        timeouts.push(timeout);
        deadlineSignal = signal;
        if (timeouts.length === 1) {
          await delay(30, undefined, { signal });
          return { ok: false, http_status: 502, error_code: "upstream_error" };
        }
        await delay(10_000, undefined, { signal });
        return { ok: true, data: [{ b64_json: "cG5n" }] };
      },
    }),
    (err) => err instanceof ImageHttpError && err.status === 504 && err.code === "worker_timeout",
  );
  assert.equal(timeouts.length, 2);
  assert(timeouts[0] <= 150 && timeouts[0] > 0);
  assert(timeouts[1] < timeouts[0] && timeouts[1] > 0);
  assert.equal(deadlineSignal.aborted, true);
  assert.deepEqual(reports.map((evt) => evt.httpStatus), [502]);
});

test("executeWebImageGeneration bounds preflight and 401 refresh waits without unauthorized accounting", async (t) => {
  for (const mode of ["preflight", "401", "cancel"]) {
    await t.test(mode, async () => {
      const account = { email: "refresh@example.com", accessToken: jwt(mode === "preflight" ? -60 : 3600) };
      const ac = new AbortController();
      const reports = [];
      let refreshOptions;
      let refreshCleaned = false;
      let workerCalls = 0;
      await assert.rejects(
        executeWebImageGeneration({
          accounts: [account],
          request: { model: IMAGE_MODEL_ID, prompt: "cat", n: 1 },
          signal: ac.signal,
          maxTotalMs: mode === "cancel" ? 5_000 : 40,
          proxy: "http://proxy.invalid:8080",
          onAccountResult: (evt) => reports.push(evt),
          reauthenticate: async (_account, options) => {
            refreshOptions = options;
            if (mode === "cancel") ac.abort();
            try {
              await delay(10_000, undefined, { signal: options.signal });
            } finally {
              await delay(10);
              refreshCleaned = true;
            }
          },
          runWorker: async () => {
            workerCalls++;
            return { ok: false, http_status: 401, error_code: "unauthorized" };
          },
        }),
        (err) => mode === "cancel" ? err.name === "AbortError" :
          err instanceof ImageHttpError && err.status === 504 && err.code === "worker_timeout",
      );
      assert.equal(workerCalls, mode === "preflight" ? 0 : 1);
      assert.equal(refreshOptions.proxy, "http://proxy.invalid:8080");
      assert.equal(refreshOptions.signal.aborted, true);
      assert.equal(refreshCleaned, true);
      assert.deepEqual(reports, []);
    });
  }
});

test("executeWebImageGeneration waits for cancelled reauthentication cleanup before settling", async (t) => {
  for (const mode of ["cancel", "deadline"]) {
    await t.test(mode, async () => {
      const ac = new AbortController();
      let startRefresh;
      const refreshStarted = new Promise((resolve) => { startRefresh = resolve; });
      let observeCleanup;
      const cleanupStarted = new Promise((resolve) => { observeCleanup = resolve; });
      let finishCleanup;
      const cleanup = new Promise((resolve) => { finishCleanup = resolve; });
      let cleanupComplete = false;
      let requestSettled = false;
      const pending = executeWebImageGeneration({
        accounts: [{ email: "cleanup@example.com", accessToken: jwt(-60) }],
        request: { model: IMAGE_MODEL_ID, prompt: "cat" },
        signal: ac.signal,
        maxTotalMs: mode === "deadline" ? 40 : 5_000,
        reauthenticate: async (_account, { signal }) => {
          startRefresh();
          try {
            await delay(10_000, undefined, { signal });
          } finally {
            observeCleanup();
            await cleanup;
            cleanupComplete = true;
          }
        },
      });
      const outcome = pending.then(
        (value) => { requestSettled = true; return { value }; },
        (error) => { requestSettled = true; return { error }; },
      );
      try {
        await refreshStarted;
        if (mode === "cancel") ac.abort();
        await cleanupStarted;
        await delay(0);
        assert.equal(requestSettled, false);
        finishCleanup();
        const { error } = await outcome;
        if (mode === "cancel") {
          assert.equal(error?.name, "AbortError");
        } else {
          assert(error instanceof ImageHttpError);
          assert.equal(error.status, 504);
          assert.equal(error.code, "worker_timeout");
        }
        assert.equal(cleanupComplete, true);
      } finally {
        ac.abort();
        finishCleanup();
        await outcome;
      }
    });
  }
});

test("executeWebImageGeneration does not start queued reauthentication after cancellation", async () => {
  const ac = new AbortController();
  let refreshCalls = 0;
  let workerCalls = 0;
  const pending = executeWebImageGeneration({
    accounts: [{ email: "queued@example.com", accessToken: jwt(-60) }],
    request: { model: IMAGE_MODEL_ID, prompt: "cat" },
    signal: ac.signal,
    reauthenticate: async (account) => { refreshCalls++; return account; },
    runWorker: async () => { workerCalls++; return { ok: true, data: [{ b64_json: "cG5n" }] }; },
  });
  ac.abort();
  await assert.rejects(pending, { name: "AbortError" });
  assert.equal(refreshCalls, 0);
  assert.equal(workerCalls, 0);
});

test("executeWebImageGeneration returns worker timeouts directly without account failover", async () => {
  const account = { email: "timeout@example.com", accessToken: jwt(3600) };
  let calls = 0;
  await assert.rejects(executeWebImageGeneration({
    accounts: [account, { ...account, email: "other@example.com" }],
    request: { model: IMAGE_MODEL_ID, prompt: "cat" },
    runWorker: async () => {
      calls++;
      return { ok: false, http_status: 504, error_code: "worker_timeout" };
    },
  }), (err) => err instanceof ImageHttpError && err.status === 504 && err.code === "worker_timeout");
  assert.equal(calls, 1);
});

test("executeWebImageGeneration preserves rejected account HTTP status and rate-limit retry windows", async () => {
  for (const [status, retryAfter] of [[401, null], [402, null], [403, null], [429, 1200], [429, 0], [502, null]]) {
    await assert.rejects(executeWebImageGeneration({
      accounts: [{ email: "rejected@example.com", accessToken: jwt(3600) }],
      request: { model: IMAGE_MODEL_ID, prompt: "cat" },
      reauthenticate: async (account) => account,
      runWorker: async () => ({ ok: false, http_status: status, error_code: "account_rejected", retry_after: retryAfter }),
    }), (err) => err instanceof ImageHttpError && err.status === status && err.code === "account_rejected" &&
      err.retryAfter === retryAfter);
  }
});

test("executeWebImageGeneration shares one deadline across all four image tasks", async () => {
  const signals = new Set();
  const timeouts = [];
  let settled = 0;
  await assert.rejects(executeWebImageGeneration({
    accounts: [{ email: "multi@example.com", accessToken: jwt(3600) }],
    request: { model: IMAGE_MODEL_ID, prompt: "cat", n: 4 },
    maxTotalMs: 40,
    runWorker: async (_payload, { signal, timeout }) => {
      signals.add(signal);
      timeouts.push(timeout);
      try {
        await delay(10_000, undefined, { signal });
      } finally {
        settled++;
      }
    },
  }), (err) => err instanceof ImageHttpError && err.status === 504 && err.code === "worker_timeout");
  assert.equal(signals.size, 1);
  assert.equal(timeouts.length, 4);
  assert(timeouts.every((timeout) => timeout > 0 && timeout <= 40));
  assert.equal(settled, 4);
});

test("executeWebImageGeneration cancels siblings and waits for their settlement before returning the first failure", async () => {
  const accounts = ["first", "sibling"].map((name) => ({
    email: `${name}@example.com`, accessToken: jwt(3600, `${name}@example.com`),
  }));
  let startSibling;
  const siblingStarted = new Promise((resolve) => { startSibling = resolve; });
  let abortSibling;
  const siblingAborted = new Promise((resolve) => { abortSibling = resolve; });
  let finishCleanup;
  const cleanup = new Promise((resolve) => { finishCleanup = resolve; });
  let siblingSettled = false;
  let requestSettled = false;
  const pending = executeWebImageGeneration({
    accounts,
    request: { model: IMAGE_MODEL_ID, prompt: "cat", n: 2 },
    runWorker: async (payload, { signal }) => {
      if (payload.access_token === accounts[0].accessToken) {
        await siblingStarted;
        return { ok: false, http_status: 400, error_code: "content_policy_violation" };
      }
      startSibling();
      await new Promise((resolve) => {
        signal.addEventListener("abort", () => { abortSibling(); resolve(); }, { once: true });
      });
      await cleanup;
      siblingSettled = true;
      throw Object.assign(new Error(), { name: "AbortError" });
    },
  });
  const outcome = assert.rejects(pending, (err) => err.status === 400 && err.code === "content_policy_violation");
  pending.then(() => { requestSettled = true; }, () => { requestSettled = true; });
  try {
    await siblingAborted;
    await delay(0);
    assert.equal(requestSettled, false);
    finishCleanup();
    await outcome;
    assert.equal(siblingSettled, true);
  } finally {
    finishCleanup();
    await outcome;
  }
});

test("pool-stats persists atomically without raw emails, JWTs or URLs and tracks cooldowns", async () => {
  const dir = await mkdtemp(join(tmpdir(), "gptfree-stats-"));
  const statsFile = join(dir, "pool-stats.json");
  const email = "secret-owner@example.com";
  const secretJwt = jwt(3600, email);
  try {
    await recordQuotaCheck({
      email,
      quota: 9,
      restoreAt: null,
      hasImageLimit: true,
      planType: "free",
    }, { statsFile });
    await recordQuotaCheck({
      email: "broken@example.com",
      error: `Failed fetching https://chatgpt.com/backend-api?tok=${secretJwt} for ${email}`,
    }, { statsFile });

    const raw = await readFile(statsFile, "utf8");
    assert(!raw.includes(email));
    assert(!raw.includes("broken@example.com"));
    assert(!raw.includes(secretJwt));
    assert(!raw.includes("https://chatgpt.com"));
    assert(raw.includes(accountSlotTag(email)));

    const stats = await readPoolStats({ statsFile });
    const rows = [
      { email, accessToken: secretJwt },
      { email: "broken@example.com", accessToken: jwt(3600, "broken@example.com") },
      { email: "unchecked@example.com", accessToken: jwt(3600, "unchecked@example.com") },
    ];
    const summary = summarizePoolStats(rows, stats);
    assert.equal(summary.quotaStats.total, 3);
    assert.equal(summary.quotaStats.active, 1);
    assert.equal(summary.quotaStats.errored, 1);
    assert.equal(summary.quotaStats.unchecked, 1);
    assert.equal(summary.quotaStats.totalQuota, 9);
    assert.equal(summary.accounts[0].planType, "free");
    assert.equal(summary.accounts[1].quotaStatus, "error");
    assert(!JSON.stringify(summary).includes("secret-owner@example.com"));

    // Rate-limit expiry behavior
    const pastIso = new Date(Date.now() - 60_000).toISOString();
    const futureIso = new Date(Date.now() + 60_000).toISOString();
    const store = createInMemoryStatsStore();
    await recordQuotaCheck({ email, quota: 0, restoreAt: pastIso }, { statsStore: store });
    const s1 = await readPoolStats({ statsStore: store });
    assert.equal(isAccountRateLimited(email, s1), false);
    await recordQuotaCheck({ email, quota: 0, restoreAt: futureIso }, { statsStore: store });
    const s2 = await readPoolStats({ statsStore: store });
    assert.equal(isAccountRateLimited(email, s2), true);
    assert(!sanitizeStatsMessage("err https://example.com user@test.com").includes("https://"));
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});

test("probeAccountImageQuota & refreshPoolAccount probe quota with 401 reauth and respect quotaOnly", async () => {
  const acc = { email: "probe@example.com", password: "p", totp: seed, accessToken: jwt(3600, "probe@example.com") };
  const renewedToken = jwt(3600, "probe@example.com") + "fresh";
  const store = createInMemoryStatsStore();

  let workerCalls = 0;
  const probeRes = await probeAccountImageQuota(acc, {
    statsStore: store,
    reauthenticate: async (a) => ({ ...a, accessToken: renewedToken }),
    runWorker: async (payload) => {
      assert.equal(payload.action, "check_image_quota");
      workerCalls++;
      if (workerCalls === 1) {
        return { ok: false, http_status: 401, error_code: "unauthorized", error: "expired" };
      }
      assert.equal(payload.access_token, renewedToken);
      return { ok: true, quota: 14, restore_at: null, has_image_limit: true, plan_type: "plus" };
    },
  });
  assert.equal(probeRes.ok, true);
  assert.equal(probeRes.renewed, true);
  assert.equal(probeRes.imageQuota, 14);
  assert.equal(probeRes.planType, "plus");

  const direct = await checkWebImageQuota({ accessToken: "" });
  assert.equal(direct.ok, false);
  assert.equal(direct.http_status, 401);

  let verifyCalls = 0;
  const quotaOnlyRes = await refreshPoolAccount(
    { ...acc, accessToken: renewedToken },
    {
      quotaOnly: true,
      statsStore: store,
      verify: async () => { verifyCalls++; return { accepted: true }; },
      runWorker: async () => ({ ok: true, quota: 11, restore_at: null, has_image_limit: true, plan_type: "free" }),
    },
  );
  assert.equal(verifyCalls, 0);
  assert.equal(quotaOnlyRes.quotaOk, true);
  assert.equal(quotaOnlyRes.imageQuota, 11);
});

test("native image routing preserves account refusal, cooldown and per-account statistics", async () => {
  const accA = { email: "slotA@example.com", password: "p", totp: seed, accessToken: jwt(3600, "slotA@example.com") };
  const accB = { email: "slotB@example.com", password: "p", totp: seed, accessToken: jwt(3600, "slotB@example.com") };
  const renewedB = jwt(3600, "slotB@example.com") + "2";
  const slotA = accountSlotTag(accA.email);
  const store = createInMemoryStatsStore();
  const request = { method: "POST", body: JSON.stringify({ model: IMAGE_MODEL_ID, prompt: "owl" }) };
  const calls = [];
  const options = {
    read: async () => [accA, accB],
    statsStore: store,
    reauthenticate: async (account) => ({ ...account, accessToken: renewedB }),
    runWorker: async (payload) => {
      calls.push(payload.access_token);
      if (payload.access_token === accA.accessToken) {
        return { ok: false, http_status: 429, error_code: "rate_limited", retry_after: 1200 };
      }
      if (payload.access_token === accB.accessToken) {
        return { ok: false, http_status: 401, error_code: "unauthorized" };
      }
      return { ok: true, data: [{ b64_json: "cG5n", media_type: "image/png" }] };
    },
  };
  await recordQuotaCheck({ email: accA.email, quota: 5 }, { statsStore: store });
  await recordQuotaCheck({ email: accB.email, quota: 5 }, { statsStore: store });

  const limited = await poolFetch(CODEX_BASE + "/images/generations", request, { ...options, preferredSlot: slotA });
  assert.equal(limited.status, 429);
  assert.equal(limited.headers.get("retry-after"), "1200");
  assert.deepEqual(calls, [accA.accessToken]);
  const cached = await poolFetch(CODEX_BASE + "/images/generations", request, { ...options, preferredSlot: slotA });
  assert.equal(cached.status, 429);
  assert(Number(cached.headers.get("retry-after")) > 0);
  assert.deepEqual(calls, [accA.accessToken]);

  // A separate Magpie attempt owns B's request, renewal and success.
  const succeeded = await poolFetch(CODEX_BASE + "/images/generations", request, {
    ...options, preferredSlot: accountSlotTag(accB.email),
  });
  assert.equal(succeeded.status, 200);
  assert.deepEqual(calls, [accA.accessToken, accB.accessToken, renewedB]);
  const summary = summarizePoolStats([accA, accB], await readPoolStats({ statsStore: store }));
  assert.equal(summary.quotaStats.limited, 1);
  assert.equal(summary.quotaStats.active, 1);
  assert.equal(summary.quotaStats.totalQuota, 4);
  assert.equal(summary.quotaStats.totalSuccess, 1);
  assert.equal(summary.quotaStats.totalFail, 1);
});

test("an upstream 403 rests its slot, answers locally while resting and ends on the next drawing", async () => {
  const account = { email: "refused@example.com", accessToken: jwt(3600, "refused@example.com") };
  const slot = accountSlotTag(account.email);
  const statsStore = createInMemoryStatsStore();
  const refusal = { ok: false, http_status: 403, error_code: "forbidden", error: "Forbidden" };
  const outcomes = [refusal, refusal, { ok: true, data: [{ b64_json: "cG5n", media_type: "image/png" }] }];
  let calls = 0;
  const draw = () => poolFetch(CODEX_BASE + "/images/generations", {
    method: "POST", body: JSON.stringify({ model: IMAGE_MODEL_ID, prompt: "owl" }),
  }, {
    read: async () => [account], preferredSlot: slot, statsStore,
    runWorker: async () => outcomes[calls++],
  });
  const summarize = async () => summarizePoolStats([account], await readPoolStats({ statsStore }));
  const endRest = () => {
    statsStore.state.accounts[slot].forbiddenUntil = new Date(Date.now() - 1000).toISOString();
  };

  assert.equal((await draw()).status, 403);
  assert.equal(calls, 1);
  let summary = await summarize();
  assert.equal(summary.quotaStats.forbidden, 1);
  assert.equal(summary.accounts[0].quotaStatus, "forbidden");
  assert.equal(summary.accounts[0].forbiddenResting, true);

  const resting = await draw();
  assert.equal(resting.status, 403);
  assert.equal((await resting.json()).error.code, "forbidden");
  const retryAfter = Number(resting.headers.get("retry-after"));
  assert(retryAfter > 0 && retryAfter <= 60);
  assert.equal(calls, 1);

  endRest();
  assert.equal((await draw()).status, 403);
  assert.equal(calls, 2);
  summary = await summarize();
  assert.equal(summary.accounts[0].forbiddenCount, 2);
  const restMs = Date.parse(summary.accounts[0].forbiddenUntil) - Date.now();
  assert(restMs > 60_000 && restMs <= 120_000);

  endRest();
  assert.equal((await draw()).status, 200);
  assert.equal(calls, 3);
  summary = await summarize();
  assert.equal(summary.quotaStats.forbidden, 0);
  assert.equal(summary.accounts[0].forbiddenCount, 0);
  assert.notEqual(summary.accounts[0].quotaStatus, "forbidden");
});

test("a slot with an expired or unknown quota reset can recover by trying upstream", async () => {
  const account = { email: "recover@example.com", accessToken: jwt(3600) };
  const statsStore = createInMemoryStatsStore();
  let calls = 0;
  for (const restoreAt of [new Date(Date.now() - 1000).toISOString(), null]) {
    await recordQuotaCheck({ email: account.email, quota: 0, restoreAt }, { statsStore });
    const response = await poolFetch(CODEX_BASE + "/images/generations", {
      body: JSON.stringify({ model: IMAGE_MODEL_ID, prompt: "owl" }),
    }, {
      read: async () => [account], preferredSlot: accountSlotTag(account.email), statsStore,
      runWorker: async () => { calls++; return { ok: true, data: [{ b64_json: "cG5n" }] }; },
    });
    assert.equal(response.status, 200);
  }
  assert.equal(calls, 2);
});
