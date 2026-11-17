/**
 * Standalone temporary access-token pool for ChatGPT's Codex backend.
 * This does NOT modify Magpie's native Codex accounts, and does not
 * pretend that access-only tokens have a refresh token.
 *
 * An access token is only admitted after OpenAI's own /codex/models endpoint
 * returns a non-empty model list. JWT exp is just a preflight hint.
 */
import { accessTokenState } from "./credentials.mjs";
import { accountSlotTag } from "./pool-connection.mjs";
import { listPool, upsertPool } from "./pool-vault.mjs";
import { loginHTTPAccount } from "./dsh-http-auth.mjs";
import {
  buildWebTextModelCatalog,
  executeWebTextGeneration,
} from "./chatgpt-web-text.mjs";
import {
  ImageHttpError,
  buildImageModelCatalog,
  checkWebImageQuota,
  executeWebImageGeneration,
  isImageModelId,
  parseImageRequest,
} from "./chatgpt-web-image.mjs";
import {
  defaultStatsFile,
  isAccountRateLimited,
  readPoolStats,
  recordImageCall,
  recordQuotaCheck,
  sanitizeStatsMessage,
} from "./pool-stats.mjs";

export const CODEX_BASE = "https://chatgpt.com/backend-api/codex";
export const CODEX_VERSION = "0.164.0";
const MODEL_URL = CODEX_BASE + "/models?client_version=" + encodeURIComponent(CODEX_VERSION);
const MODEL_ID = /^[a-zA-Z0-9._:-]{1,128}$/;
const modelPrompts = new Map();
const pendingAuth = new Map();
let nextIndex = 0;
let authQueue = Promise.resolve();

function accessClaims(token) {
  try {
    const payload = String(token).split(".")[1];
    if (!payload || payload.length > 65536) return {};
    return JSON.parse(Buffer.from(payload, "base64url").toString("utf8"));
  } catch { return {}; }
}

function getAccountID(token) {
  const c = accessClaims(token);
  return c?.["https://api.openai.com/auth"]?.chatgpt_account_id ||
    c?.chatgpt_account_id || "";
}

export function isWebOnlyToken(token) {
  const claims = accessClaims(token);
  if (!claims) return false;
  if (getAccountID(token)) return false;
  return String(claims.client_id || "") === "app_2SKx67EdpoN0G6j64rFvigXD"
    || Boolean(claims["https://api.openai.com/auth"] || claims["https://api.openai.com/profile"]);
}

export function isCodexToken(token) {
  return !isWebOnlyToken(token);
}

function upstreamHeaders(token, accountID, streaming) {
  const h = new Headers({
    Authorization: "Bearer " + token,
    originator: "magpie_gptfree",
    "User-Agent": `MagpieGPTFree/${CODEX_VERSION}`,
    version: CODEX_VERSION,
    "OpenAI-Beta": "responses=experimental",
    Accept: streaming ? "text/event-stream" : "application/json",
  });
  if (accountID) h.set("chatgpt-account-id", accountID);
  if (streaming) h.set("Content-Type", "application/json");
  return h;
}

export function parseCodexCatalog(payload) {
  if (!Array.isArray(payload?.models)) throw new Error("Codex models response is invalid");
  const models = {};
  for (const item of payload.models.slice(0, 256)) {
    const id = item?.slug;
    if (typeof id !== "string" || !MODEL_ID.test(id) ||
        (item.visibility && item.visibility !== "list")) continue;
    const title = typeof item.display_name === "string" ? item.display_name : id;
    const prompt = item?.model_messages?.instructions_template || item?.base_instructions;
    if (typeof prompt === "string" && prompt.length <= 150000) modelPrompts.set(id, prompt);
    models[id] = {
      id, name: title,
      api: { id, url: CODEX_BASE, npm: "@ai-sdk/openai" },
      provider: { npm: "@ai-sdk/openai" },
      tool_call: true,
      limit: { context: Number(item.context_window) || 200000, output: 32000 },
    };
  }
  if (!Object.keys(models).length) throw new Error("Codex did not provide usable models");
  return models;
}

export async function verifyAccessToken(token, { fetcher = fetch, signal, expectedEmail = "" } = {}) {
  if (accessTokenState(token).state !== "unexpired_unverified")
    return { accepted: false, reason: "expired_or_unverified" };
  const claims = accessClaims(token);
  const who = String(claims.email || "").trim().toLowerCase();
  if (who && expectedEmail && who !== expectedEmail.trim().toLowerCase())
    return { accepted: false, reason: "identity_mismatch" };
  const h = upstreamHeaders(token, getAccountID(token), false);
  // decision: compose cancellation with AbortController; not AbortSignal.any
  // because supported Node 20.0–20.2 releases do not provide it.
  const controller = new AbortController();
  const forwardAbort = () => controller.abort(signal.reason);
  const timer = setTimeout(() => controller.abort(new DOMException("Token verification timed out", "TimeoutError")), 15000);
  timer.unref?.();
  signal?.addEventListener("abort", forwardAbort, { once: true });
  if (signal?.aborted) forwardAbort();
  try {
    let response;
    try {
      response = await fetcher(MODEL_URL, {
        headers: h, method: "GET", redirect: "error", signal: controller.signal,
      });
    } catch {
      if (signal?.aborted) throw signal.reason;
      return { accepted: false, reason: "network_error" };
    }
    if (!response.ok) return { accepted: false, reason: "status_" + response.status };
    try {
      const models = parseCodexCatalog(await response.json());
      return { accepted: true, models };
    } catch {
      if (signal?.aborted) throw signal.reason;
      return { accepted: false, reason: "invalid_catalog" };
    }
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener("abort", forwardAbort);
  }
}

export function prepareCodexBody(raw) {
  let b;
  try { b = typeof raw === "string" ? JSON.parse(raw) : structuredClone(raw); }
  catch { throw new Error("Cannot decode Responses request body"); }
  if (!b || typeof b !== "object" || Array.isArray(b))
    throw new Error("Responses request body is not an object");
  for (const key of [
    "max_output_tokens", "max_completion_tokens", "temperature", "top_p",
    "previous_response_id", "user", "safety_identifier", "prompt_cache_options",
  ]) delete b[key];
  if (!["priority", "ultrafast"].includes(b.service_tier)) delete b.service_tier;
  if (typeof b.input === "string")
    b.input = [{ type: "message", role: "user", content: [{ type: "input_text", text: b.input }] }];
  if (!Array.isArray(b.input)) b.input = [];
  b.input = b.input.filter((item) => item?.type !== "item_reference").map((item) => {
    if (!item || typeof item !== "object") return item;
    const copy = { ...item };
    delete copy.id;
    if (["message", "reasoning"].includes(copy.type)) delete copy.status;
    if (copy.role === "system") copy.role = "developer";
    return copy;
  });
  const own = typeof b.instructions === "string" ? b.instructions.trim() : "";
  const standard = modelPrompts.get(b.model);
  if (standard && own && standard !== own) {
    b.input.unshift({
      type: "message", role: "developer",
      content: [{ type: "input_text", text: own }],
    });
    b.instructions = standard;
  } else if (standard) b.instructions = standard;
  b.store = false;
  b.stream = true;
  b.tool_choice = "auto";
  if (b.parallel_tool_calls === undefined) b.parallel_tool_calls = true;
  if (!b.text) b.text = { verbosity: "medium" };
  if (b.reasoning?.effort === "ultra") b.reasoning.effort = "max";
  if (b.reasoning?.effort !== "none") {
    if (b.reasoning && !b.reasoning.summary) b.reasoning.summary = "auto";
    if (!Array.isArray(b.include)) b.include = [];
    if (!b.include.includes("reasoning.encrypted_content"))
      b.include.push("reasoning.encrypted_content");
  }
  return JSON.stringify(b);
}

export async function renewAccount(account, {
  authenticate = loginHTTPAccount, verify = verifyAccessToken, save = upsertPool, proxy = "", signal,
} = {}) {
  signal?.throwIfAborted();
  let pending = pendingAuth.get(account.email);
  if (!pending || pending.controller.signal.aborted) {
    pending = { controller: new AbortController(), waiters: 0, started: false, settled: false, task: null };
    const entry = pending;
    const renewalSignal = entry.controller.signal;
    // DSH HTTP login uses isolated workers; serialize renewals to avoid a surge
    // of simultaneous reauthorization attempts.
    entry.task = authQueue.then(async () => {
      renewalSignal.throwIfAborted();
      entry.started = true;
      const { tokens } = await authenticate(account, { proxy, signal: renewalSignal });
      renewalSignal.throwIfAborted();
      const verified = await verify(tokens.access, { expectedEmail: account.email, signal: renewalSignal });
      renewalSignal.throwIfAborted();
      if (!verified.accepted)
        throw new Error("HTTP 登录返回的 Platform Token 不能用于当前 Magpie 模型接口");
      const record = { ...account, accessToken: tokens.access };
      await save(record);
      return record;
    }).finally(() => {
      entry.settled = true;
      if (pendingAuth.get(account.email) === entry) pendingAuth.delete(account.email);
    });
    authQueue = entry.task.catch(() => {});
    pendingAuth.set(account.email, entry);
  }
  // decision: cancel a shared renewal only after its last caller leaves; not
  // bind it to the first caller's signal because other requests may still need it.
  pending.waiters++;
  let onAbort;
  try {
    if (!signal) return await pending.task;
    return await new Promise((resolve, reject) => {
      onAbort = () => reject(signal.reason);
      signal.addEventListener("abort", onAbort, { once: true });
      if (signal.aborted) onAbort();
      pending.task.then(resolve, reject);
    });
  } finally {
    if (onAbort) signal.removeEventListener("abort", onAbort);
    pending.waiters--;
    if (!pending.waiters && !pending.settled) {
      pending.controller.abort();
      if (pending.started) await pending.task.catch(() => {});
    }
  }
}

async function candidate(account, { reauthenticate = renewAccount, proxy = "", signal } = {}) {
  signal?.throwIfAborted();
  if (accessTokenState(account.accessToken).state === "unexpired_unverified")
    return account;
  return reauthenticate(account, { proxy, signal });
}

function orderAccountsForSlot(accounts, preferredSlot = "", rotate = false) {
  if (!Array.isArray(accounts) || !accounts.length) return [];
  if (typeof preferredSlot === "string" && preferredSlot) {
    // decision: keep a Magpie request within its selected account; not silently
    // fail over here because Magpie owns account toggles, proxies and usage.
    return accounts.filter((a) => typeof a?.email === "string" && accountSlotTag(a.email) === preferredSlot);
  }
  if (!rotate) return accounts.slice();
  const start = nextIndex++ % accounts.length;
  return [...accounts.slice(start), ...accounts.slice(0, start)];
}

function mergeDiscoveredModels(models, token) {
  return {
    ...(isWebOnlyToken(token) ? buildWebTextModelCatalog(CODEX_BASE) : {}),
    ...(models || {}),
    ...buildImageModelCatalog(CODEX_BASE),
  };
}

export async function discoverPoolModels({
  read = listPool, fetcher = fetch, reauthenticate = renewAccount, preferredSlot = "",
} = {}) {
  const accounts = orderAccountsForSlot(await read(), preferredSlot, false);
  if (!accounts.length) return {};
  let fallbackWebOnly = false;
  for (const account of accounts) {
    if (accessTokenState(account.accessToken).state !== "unexpired_unverified")
      continue;
    const probe = await verifyAccessToken(account.accessToken, { fetcher });
    if (probe.accepted) return mergeDiscoveredModels(probe.models, account.accessToken);
    if (isWebOnlyToken(account.accessToken) && probe.status !== 401) {
      fallbackWebOnly = true;
    }
  }
  for (const account of accounts) {
    try {
      if (reauthenticate === renewAccount) {
        let models = null;
        let acceptedToken = "";
        await renewAccount(account, {
          verify: async (token, opts) => {
            const res = await verifyAccessToken(token, { ...opts, fetcher });
            if (res.accepted) {
              models = res.models;
              acceptedToken = token;
            }
            return res;
          },
        });
        if (models) return mergeDiscoveredModels(models, acceptedToken);
      } else {
        const renewed = await reauthenticate(account);
        if (renewed?.accessToken) {
          const probe = await verifyAccessToken(renewed.accessToken, { fetcher });
          if (probe.accepted) return mergeDiscoveredModels(probe.models, renewed.accessToken);
          if (isWebOnlyToken(renewed.accessToken) && probe.status !== 401) {
            fallbackWebOnly = true;
          }
        }
      }
    } catch {
      // Try the next pooled account before giving up on discovery.
    }
  }
  if (fallbackWebOnly) {
    return { ...buildWebTextModelCatalog(CODEX_BASE), ...buildImageModelCatalog(CODEX_BASE) };
  }
  return {};
}

function friendlyRefreshReason(err) {
  const message = String(err?.message || "");
  const known = /DSH HTTP|Platform Token|没有返回可用|未找到可用 Python|HTTP 登录|认证后端|不能用于当前 Magpie/.test(message);
  return known ? sanitizeStatsMessage(message, 180) :
    "检测或刷新授权失败；请检查账号、TOTP、网络或代理";
}

function friendlyQuotaError(res) {
  const status = Number(res?.http_status || 502);
  const code = String(res?.error_code || "");
  if (status === 401 || code === "unauthorized") {
    return "网页鉴权 401（Token 无 chatgpt.com 网页权限或已失效）";
  }
  if (status === 403 || code === "forbidden") {
    return "Cloudflare / 网页端拒绝访问 (403)，请尝试配置或更换代理";
  }
  if (status === 504 || code === "worker_timeout" || code === "timeout") {
    return "查询生图额度超时，请检查网络或 HTTP(S) 代理";
  }
  if (code === "dependencies_missing" || code === "worker_start_failed") {
    return "Python 生图 Worker 环境不可用";
  }
  return sanitizeStatsMessage(res?.error) || "查询生图额度失败，请检查网络或代理";
}

export async function probeAccountImageQuota(account, {
  proxy = "",
  timezoneOffsetMin = -480,
  runWorker,
  reauthenticate = renewAccount,
  statsFile,
  statsStore,
  signal,
} = {}) {
  const targetStats = {
    ...(statsFile !== undefined ? { statsFile } : {}),
    ...(statsStore !== undefined ? { statsStore } : {}),
  };
  if (!account || typeof account.email !== "string" || !account.email.includes("@")) {
    return {
      ok: false,
      imageQuota: null,
      restoreAt: null,
      hasImageLimit: null,
      planType: null,
      rateLimited: false,
      error: "账号记录无效",
      httpStatus: 400,
      errorCode: "invalid_account",
      renewed: false,
      account,
    };
  }

  let current = account;
  let renewed = false;
  if (accessTokenState(current?.accessToken).state !== "unexpired_unverified") {
    if (typeof reauthenticate === "function") {
      try {
        current = await reauthenticate(current, { proxy });
        renewed = true;
      } catch (err) {
        const msg = friendlyRefreshReason(err);
        await recordQuotaCheck({ email: account.email, error: msg }, targetStats);
        return {
          ok: false,
          imageQuota: null,
          restoreAt: null,
          hasImageLimit: null,
          planType: null,
          rateLimited: false,
          error: msg,
          httpStatus: 401,
          errorCode: "unauthorized",
          renewed: false,
          account,
        };
      }
    } else {
      const msg = "Token 已过期或缺失，需要重新授权";
      await recordQuotaCheck({ email: account.email, error: msg }, targetStats);
      return {
        ok: false,
        imageQuota: null,
        restoreAt: null,
        hasImageLimit: null,
        planType: null,
        rateLimited: false,
        error: msg,
        httpStatus: 401,
        errorCode: "unauthorized",
        renewed: false,
        account,
      };
    }
  }

  let res = await checkWebImageQuota({
    accessToken: current.accessToken,
    proxy,
    timezoneOffsetMin,
    signal,
    ...(runWorker ? { runWorker } : {}),
  });

  if (res?.ok !== true && Number(res?.http_status) === 401 && typeof reauthenticate === "function") {
    try {
      const updated = await reauthenticate(current, { proxy });
      if (updated?.accessToken) {
        current = updated;
        renewed = true;
        res = await checkWebImageQuota({
          accessToken: current.accessToken,
          proxy,
          timezoneOffsetMin,
          signal,
          ...(runWorker ? { runWorker } : {}),
        });
      }
    } catch (err) {
      const msg = friendlyRefreshReason(err);
      await recordQuotaCheck({ email: account.email, error: msg }, targetStats);
      return {
        ok: false,
        imageQuota: null,
        restoreAt: null,
        hasImageLimit: null,
        planType: null,
        rateLimited: false,
        error: msg,
        httpStatus: 401,
        errorCode: "unauthorized",
        renewed,
        account: current,
      };
    }
  }

  if (res?.ok === true) {
    const imageQuota = Math.max(0, Math.trunc(Number(res.quota ?? 0) || 0));
    const restoreAt = res.restore_at || null;
    const hasImageLimit = typeof res.has_image_limit === "boolean" ? res.has_image_limit : null;
    const planType = res.plan_type || null;
    await recordQuotaCheck({
      email: account.email,
      quota: imageQuota,
      restoreAt,
      hasImageLimit,
      planType,
    }, targetStats);
    return {
      ok: true,
      imageQuota,
      restoreAt,
      hasImageLimit,
      planType,
      rateLimited: imageQuota === 0,
      error: null,
      httpStatus: 200,
      errorCode: "",
      renewed,
      account: current,
    };
  }

  if (Number(res?.http_status) === 429 || res?.error_code === "rate_limited") {
    const updatedStats = await recordQuotaCheck({
      email: account.email,
      rateLimited: true,
      retryAfter: res?.retry_after ?? null,
      restoreAt: res?.restore_at ?? null,
    }, targetStats);
    const slotRec = updatedStats?.accounts?.[accountSlotTag(account.email)];
    return {
      ok: true,
      imageQuota: 0,
      restoreAt: slotRec?.limitedUntil || slotRec?.restoreAt || res?.restore_at || null,
      hasImageLimit: true,
      planType: slotRec?.planType || null,
      rateLimited: true,
      error: null,
      httpStatus: 429,
      errorCode: "rate_limited",
      renewed,
      account: current,
    };
  }

  const errMsg = friendlyQuotaError(res);
  await recordQuotaCheck({ email: account.email, error: errMsg }, targetStats);
  return {
    ok: false,
    imageQuota: null,
    restoreAt: null,
    hasImageLimit: null,
    planType: null,
    rateLimited: false,
    error: errMsg,
    httpStatus: Number(res?.http_status || 502),
    errorCode: String(res?.error_code || "quota_check_failed"),
    renewed,
    account: current,
  };
}

export async function refreshPoolAccount(account, {
  fetcher = fetch, verify = verifyAccessToken, reauthenticate = renewAccount,
  proxy = "", forceRenew = false,
  includeQuota = false, quotaOnly = false, timezoneOffsetMin = -480,
  runWorker, statsFile, statsStore, signal,
} = {}) {
  if (!account || typeof account.email !== "string")
    return { ok: false, status: "failed", renewed: false, reason: "账号记录无效" };

  if (!includeQuota && !quotaOnly) {
    try {
      if (!forceRenew && accessTokenState(account.accessToken).state === "unexpired_unverified") {
        const probe = await verify(account.accessToken, { fetcher, expectedEmail: account.email });
        if (probe.accepted) {
          return {
            ok: true,
            status: "valid",
            renewed: false,
            tokenState: "unexpired_unverified",
            modelCount: Object.keys(probe.models || {}).length,
          };
        }
      }
      const updated = await reauthenticate(account, { proxy });
      return {
        ok: true,
        status: "renewed",
        renewed: true,
        tokenState: accessTokenState(updated?.accessToken || "").state,
      };
    } catch (err) {
      return { ok: false, status: "failed", renewed: false, reason: friendlyRefreshReason(err) };
    }
  }

  const resolvedStats = (statsFile !== undefined || statsStore !== undefined)
    ? {
        ...(statsFile !== undefined ? { statsFile } : {}),
        ...(statsStore !== undefined ? { statsStore } : {}),
      }
    : (verify === verifyAccessToken && reauthenticate === renewAccount
      ? { statsFile: defaultStatsFile() }
      : { statsFile: null });

  if (quotaOnly) {
    let targetAccount = account;
    let forcedRenew = false;
    try {
      if (forceRenew && typeof reauthenticate === "function") {
        targetAccount = await reauthenticate(account, { proxy });
        forcedRenew = true;
      }
      const quotaProbe = await probeAccountImageQuota(targetAccount, {
        proxy,
        timezoneOffsetMin,
        runWorker,
        reauthenticate,
        ...resolvedStats,
        signal,
      });
      const wasRenewed = forcedRenew || quotaProbe.renewed;
      const tokenState = accessTokenState(quotaProbe.account?.accessToken || targetAccount?.accessToken || "").state;
      if (quotaProbe.ok) {
        return {
          ok: true,
          status: wasRenewed ? "renewed" : "valid",
          renewed: wasRenewed,
          tokenState,
          quotaOk: true,
          imageQuota: quotaProbe.imageQuota,
          quotaResetAt: quotaProbe.restoreAt,
          hasImageLimit: quotaProbe.hasImageLimit,
          planType: quotaProbe.planType,
          quotaError: null,
        };
      }
      return {
        ok: false,
        status: "failed",
        renewed: wasRenewed,
        tokenState,
        reason: quotaProbe.error || "生图额度查询失败",
        quotaOk: false,
        imageQuota: null,
        quotaResetAt: null,
        hasImageLimit: null,
        planType: null,
        quotaError: quotaProbe.error || "生图额度查询失败",
      };
    } catch (err) {
      return {
        ok: false,
        status: "failed",
        renewed: false,
        reason: friendlyRefreshReason(err),
        quotaOk: false,
        quotaError: friendlyRefreshReason(err),
      };
    }
  }

  let baseResult;
  let currentAccount = account;
  try {
    let acceptedLive = false;
    if (!forceRenew && accessTokenState(account.accessToken).state === "unexpired_unverified") {
      const probe = await verify(account.accessToken, { fetcher, expectedEmail: account.email });
      if (probe.accepted) {
        acceptedLive = true;
        baseResult = {
          ok: true,
          status: "valid",
          renewed: false,
          tokenState: "unexpired_unverified",
          modelCount: Object.keys(probe.models || {}).length,
        };
      }
    }
    if (!acceptedLive) {
      currentAccount = await reauthenticate(account, { proxy });
      baseResult = {
        ok: true,
        status: "renewed",
        renewed: true,
        tokenState: accessTokenState(currentAccount?.accessToken || "").state,
      };
    }
  } catch (err) {
    return { ok: false, status: "failed", renewed: false, reason: friendlyRefreshReason(err) };
  }

  const quotaProbe = await probeAccountImageQuota(currentAccount, {
    proxy,
    timezoneOffsetMin,
    runWorker,
    reauthenticate,
    ...resolvedStats,
    signal,
  });
  return {
    ...baseResult,
    renewed: Boolean(baseResult.renewed || quotaProbe.renewed),
    status: quotaProbe.renewed ? "renewed" : baseResult.status,
    tokenState: accessTokenState(quotaProbe.account?.accessToken || currentAccount?.accessToken || "").state,
    quotaOk: quotaProbe.ok,
    imageQuota: quotaProbe.imageQuota,
    quotaResetAt: quotaProbe.restoreAt,
    hasImageLimit: quotaProbe.hasImageLimit,
    planType: quotaProbe.planType,
    quotaError: quotaProbe.error || null,
  };
}

function jsonResponse(payload, status = 200) {
  return new Response(JSON.stringify(payload), {
    status,
    headers: { "content-type": "application/json" },
  });
}

function handleChatProbe(rawBody) {
  let model = "";
  try {
    const text = rawBody instanceof Uint8Array ? Buffer.from(rawBody).toString("utf8") :
      rawBody instanceof ArrayBuffer ? Buffer.from(rawBody).toString("utf8") :
      String(rawBody || "");
    const parsed = text ? JSON.parse(text) : {};
    model = String(parsed?.model || "").trim();
  } catch {}
  if (isImageModelId(model)) {
    return jsonResponse({
      error: {
        message: "This image model does not support /v1/chat/completions; use /v1/images/generations or /v1/images/edits instead.",
        type: "invalid_request_error",
      },
    }, 400);
  }
  return jsonResponse({
    error: {
      message: "Codex models on this provider require /v1/responses instead of /v1/chat/completions.",
      type: "invalid_request_error",
    },
  }, 400);
}

export async function poolFetch(input, init, {
  read = listPool, fetcher = fetch, reauthenticate = renewAccount, preferredSlot = "",
  runWorker, webTextRunner, proxy = "", statsFile, statsStore,
} = {}) {
  let dest;
  try { dest = new URL(String(input)); }
  catch { throw new Error("Invalid Magpie request URL"); }
  if (dest.origin !== "https://chatgpt.com" || dest.username || dest.password || dest.hash)
    throw new Error("GPTFree plugin only forwards Codex /responses and /images endpoints");

  if (dest.pathname === "/backend-api/codex/chat/completions") {
    return handleChatProbe(init?.body);
  }

  if (dest.pathname === "/backend-api/codex/images/generations" ||
      dest.pathname === "/backend-api/codex/images/edits") {
    let parsedReq;
    try {
      parsedReq = await parseImageRequest(dest.pathname, init?.body, init?.headers);
    } catch (err) {
      if (err instanceof ImageHttpError) {
        return jsonResponse({
          error: {
            message: err.message,
            type: "invalid_request_error",
            ...(err.code ? { code: err.code } : {}),
          },
        }, err.status);
      }
      return jsonResponse({ error: { message: "Invalid image request", type: "invalid_request_error" } }, 400);
    }

    const accounts = await read();
    if (!accounts.length) {
      return jsonResponse({ error: { message: "GPTFree pool is empty; import accounts first" } }, 503);
    }

    const resolvedStats = (statsFile !== undefined || statsStore !== undefined)
      ? {
          ...(statsFile !== undefined ? { statsFile } : {}),
          ...(statsStore !== undefined ? { statsStore } : {}),
        }
      : (read === listPool ? { statsFile: defaultStatsFile() } : { statsFile: null });

    const rawOrdered = orderAccountsForSlot(accounts, preferredSlot, true);
    if (!rawOrdered.length) {
      return jsonResponse({ error: { message: "Selected GPTFree account is no longer in the pool", code: "account_unavailable" } }, 401);
    }
    const stats = await readPoolStats(resolvedStats).catch(() => null);
    let retryAt = Infinity;
    const ordered = rawOrdered.filter((account) => {
      if (!stats || !isAccountRateLimited(account.email, stats)) return true;
      const quota = stats.accounts[accountSlotTag(account.email)];
      const resetAt = Date.parse(quota?.limitedUntil || quota?.restoreAt || "");
      // A quota without a reset time needs another upstream check to recover.
      if (!Number.isFinite(resetAt)) return true;
      retryAt = Math.min(retryAt, resetAt);
      return false;
    });
    if (!ordered.length) {
      const response = jsonResponse({ error: { message: "Image quota is temporarily exhausted", code: "rate_limited" } }, 429);
      response.headers.set("retry-after", String(Math.max(1, Math.ceil((retryAt - Date.now()) / 1000))));
      return response;
    }

    try {
      const result = await executeWebImageGeneration({
        accounts: ordered,
        request: parsedReq,
        signal: init?.signal,
        reauthenticate,
        ...(runWorker ? { runWorker } : {}),
        proxy,
        onAccountResult: (evt) => {
          if (!resolvedStats.statsStore && !resolvedStats.statsFile) return;
          return recordImageCall({
            email: evt.account?.email,
            success: evt.success === true,
            count: evt.count || 1,
            rateLimited: evt.rateLimited === true,
            retryAfter: evt.retryAfter ?? null,
            error: evt.errorCode || "",
          }, resolvedStats);
        },
      });
      return jsonResponse(result, 200);
    } catch (err) {
      if (err?.name === "AbortError") throw err;
      if (err instanceof ImageHttpError) {
        const response = jsonResponse({
          error: {
            message: err.message,
            type: err.status === 400 ? "invalid_request_error" : "server_error",
            ...(err.code ? { code: err.code } : {}),
          },
        }, err.status);
        const retryAfter = Number(err.retryAfter);
        if (err.retryAfter != null && Number.isFinite(retryAfter) && retryAfter >= 0) {
          response.headers.set("retry-after", String(Math.ceil(retryAfter)));
        }
        return response;
      }
      return jsonResponse({ error: { message: "Image generation failed", type: "server_error" } }, 503);
    }
  }

  if (dest.pathname !== "/backend-api/codex/responses")
    throw new Error("GPTFree plugin only forwards Codex /responses and /images endpoints");
  let rawParsed;
  try {
    rawParsed = JSON.parse(String(init?.body || "{}"));
    if (!rawParsed || typeof rawParsed !== "object" || Array.isArray(rawParsed)) {
      throw new Error("invalid body");
    }
  } catch {
    return new Response('{"error":{"message":"Invalid Responses request"}}', {
      status: 400, headers: { "content-type": "application/json" },
    });
  }
  const accounts = await read();
  if (!accounts.length) return new Response('{"error":{"message":"GPTFree pool is empty; import accounts first"}}', {
    status: 503, headers: { "content-type": "application/json" },
  });
  const ordered = orderAccountsForSlot(accounts, preferredSlot, true);
  if (!ordered.length) {
    return jsonResponse({ error: { message: "Selected GPTFree account is no longer in the pool", code: "account_unavailable" } }, 401);
  }
  const webTextAccounts = ordered.filter((a) => isWebOnlyToken(a?.accessToken));

  if (typeof webTextRunner === "function") {
    return executeWebTextGeneration(
      init?.body,
      webTextAccounts.length ? webTextAccounts : ordered,
      { reauthenticate, signal: init?.signal, worker: webTextRunner, proxy },
    );
  }

  const codexAccounts = ordered.filter((a) => !isWebOnlyToken(a?.accessToken));
  if (!codexAccounts.length) {
    return executeWebTextGeneration(init?.body, ordered, {
      reauthenticate,
      signal: init?.signal,
      proxy,
    });
  }

  let body;
  try { body = prepareCodexBody(init?.body); }
  catch { return new Response('{"error":{"message":"Invalid Responses request"}}', { status: 400, headers: { "content-type": "application/json" } }); }

  for (let idx = 0; idx < codexAccounts.length; idx++) {
    const original = codexAccounts[idx];
    let account;
    try { account = await candidate(original, { reauthenticate, proxy, signal: init?.signal }); }
    catch {
      init?.signal?.throwIfAborted();
      if (preferredSlot) return jsonResponse({ error: { message: "Selected GPTFree account could not renew its authorization", code: "reauth_failed" } }, 401);
      continue;
    }
    if (isWebOnlyToken(account.accessToken)) {
      return executeWebTextGeneration(
        init?.body,
        [account, ...codexAccounts.slice(idx + 1), ...webTextAccounts],
        { reauthenticate, signal: init?.signal, proxy },
      );
    }
    for (let attempt = 0; attempt < 2; attempt++) {
      const headers = upstreamHeaders(account.accessToken, getAccountID(account.accessToken), true);
      const key = rawParsed?.prompt_cache_key;
      if (typeof key === "string" && key.length < 256) {
        headers.set("session-id", key);
        headers.set("thread-id", key);
      }
      let r;
      try {
        r = await fetcher(CODEX_BASE + "/responses", {
          method: "POST", body, headers, redirect: "error", signal: init?.signal,
        });
      } catch {
        init?.signal?.throwIfAborted();
        break;
      }
      if (r.ok) return r;
      if (r.status === 401 && attempt === 0) {
        try {
          account = await reauthenticate(account, { proxy, signal: init?.signal });
          await r.body?.cancel();
          if (isWebOnlyToken(account.accessToken)) {
            return executeWebTextGeneration(
              init?.body,
              [account, ...codexAccounts.slice(idx + 1), ...webTextAccounts],
              { reauthenticate, signal: init?.signal, proxy },
            );
          }
          continue;
        }
        catch {
          init?.signal?.throwIfAborted();
          if (preferredSlot) return r;
        }
      }
      if (preferredSlot || ![401, 429, 500, 502, 503, 504].includes(r.status)) return r;
      await r.body?.cancel();
      break;
    }
  }
  if (webTextAccounts.length) {
    return executeWebTextGeneration(init?.body, webTextAccounts, {
      reauthenticate,
      signal: init?.signal,
      proxy,
    });
  }
  return new Response('{"error":{"message":"所有 GPT号池 账号均不可用；请检查后台 HTTP 登录、目标模型授权及账号权限。"}}', {
    status: 503, headers: { "content-type": "application/json" },
  });
}
