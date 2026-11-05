"use strict";
/* AI Learning Assistant: plain JS front end for the FastAPI backend (same origin).
   Security rule: every string that comes from the server is shown with textContent / createTextNode only. */

// ====================================================================== config
const MAX_UPLOAD_MB = 20; // must equal the backend's MAX_UPLOAD_MB
const ALLOWED_EXT = [".pdf", ".docx", ".txt"];
const POLL_MS = 2000;
const SLOW_AFTER_MS = 30000;
const STOP_AFTER_MS = 180000;
const HISTORY_PAGE = 10;
const MAX_MESSAGE_CHARS = 500;
const TIMEOUT_MS = { health: 3000, list: 10000, upload: 60000, chat: 60000, generate: 120000 };
const GENERATE_HINT = " Nếu máy chủ đã làm xong, bấm lại sẽ lấy kết quả đã lưu ngay, không tốn thêm lượt gọi mô hình.";
const OPTION_LETTERS = ["A", "B", "C", "D"];
const STATUS_TEXT = {
  400: "Yêu cầu không hợp lệ.",
  404: "Không tìm thấy dữ liệu yêu cầu.",
  409: "Thao tác không phù hợp với trạng thái hiện tại của tài liệu.",
  413: "Tệp vượt quá dung lượng tối đa cho phép.",
  415: "Định dạng tệp không được hỗ trợ.",
  422: "Dữ liệu gửi lên không hợp lệ.",
  429: "Dịch vụ đang giới hạn số lượt gọi. Hãy thử lại sau ít phút.",
  502: "Máy chủ gặp lỗi khi gọi dịch vụ phía sau. Hãy thử lại sau.",
  503: "Dịch vụ tạm thời không khả dụng. Hãy thử lại sau.",
  504: "Máy chủ phản hồi quá lâu. Hãy thử lại.",
};
const STATUS_BADGE = {
  processing: ["Đang xử lý", "warning"],
  ready: ["Sẵn sàng", "success"],
  failed: ["Lỗi", "danger"],
};
const DROP_REASONS = {
  schema_invalid: "sai định dạng",
  duplicate_options: "đáp án trùng nhau",
  positional_option: "đáp án tham chiếu đáp án khác",
  unknown_source: "dẫn nguồn không có trong các đoạn đã gửi",
  quote_not_in_source: "trích dẫn không có nguyên văn trong tài liệu",
  duplicate_question: "câu hỏi trùng",
  verifier_key_not_supported: "bộ xác minh: đáp án chưa được nguồn chứng minh",
  verifier_ambiguous: "bộ xác minh: có đáp án khác cũng đúng",
};
const SAVE_FAILED_TEXT = "Câu trả lời này chưa được lưu vào lịch sử (lỗi lưu trữ phía máy chủ). Câu trả lời vẫn dùng được; hãy chép lại nếu cần.";

// ====================================================================== api
class ApiError extends Error {
  constructor(kind, message, extra) {
    super(message);
    this.kind = kind; // 'unreachable' | 'timeout' | 'bad_response' | 'http'
    this.message = message;
    this.status = (extra && extra.status) || null;
    this.code = (extra && extra.code) || null;
  }
}

function clip(text) {
  return text.length <= MAX_MESSAGE_CHARS ? text : text.slice(0, MAX_MESSAGE_CHARS - 1) + "…";
}

function httpErrorFrom(response, body) {
  const status = response.status;
  let message = null;
  let code = null;
  const error = body && typeof body === "object" ? body.error : null;
  if (error && typeof error.message === "string" && error.message.trim()) {
    message = clip(error.message.trim());
    code = typeof error.code === "string" ? error.code : null;
  }
  if (message === null) {
    message = STATUS_TEXT[status] ||
      (status >= 500 ? "Máy chủ gặp lỗi (mã " + status + "). Hãy thử lại sau." : "Yêu cầu bị từ chối (mã " + status + ").");
  }
  if (status === 429) {
    const retry = parseInt(response.headers.get("Retry-After"), 10);
    if (retry > 0) message += " (Máy chủ gợi ý chờ khoảng " + retry + " giây.)";
  }
  return new ApiError("http", message, { status, code });
}

async function request(method, path, opts) {
  const options = opts || {};
  const timeoutMs = options.timeout || TIMEOUT_MS.list;
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeoutMs);
  const init = { method, signal: controller.signal, headers: {} };
  if (options.json !== undefined) {
    init.headers["Content-Type"] = "application/json";
    init.body = JSON.stringify(options.json);
  } else if (options.form) {
    init.body = options.form;
  }
  let response;
  try {
    response = await fetch(path, init);
  } catch (err) {
    if (err && err.name === "AbortError") {
      throw new ApiError("timeout",
        "Máy chủ chưa trả lời sau " + timeoutMs / 1000 + " giây. Yêu cầu có thể vẫn đang được xử lý ở phía máy chủ, " +
        "nên chưa chắc nó đã thất bại. Hãy đợi một chút, kiểm tra lại danh sách tài liệu rồi thử lại." + (options.hint || ""));
    }
    throw new ApiError("unreachable",
      "Không kết nối được tới máy chủ API. Hãy kiểm tra máy chủ đang chạy (uvicorn app.main:app) rồi thử lại.");
  } finally {
    clearTimeout(timer);
  }
  let body = null;
  try {
    body = await response.json();
  } catch (err) {
    body = null;
  }
  if (!response.ok) throw httpErrorFrom(response, body);
  if (body === null) throw new ApiError("bad_response", "Máy chủ trả về dữ liệu không đọc được. Hãy thử lại sau.", { status: response.status });
  return { data: body, status: response.status };
}

const api = {
  health: () => request("GET", "/health", { timeout: TIMEOUT_MS.health }),
  listDocuments: () => request("GET", "/documents"),
  upload: (file) => {
    const form = new FormData();
    form.append("file", file, file.name);
    return request("POST", "/documents", { form, timeout: TIMEOUT_MS.upload });
  },
  deleteDocument: (id) => request("DELETE", "/documents/" + encodeURIComponent(id)),
  chat: (body) => request("POST", "/chat", { json: body, timeout: TIMEOUT_MS.chat }),
  summary: (id, style, refresh) => request("POST", "/documents/" + encodeURIComponent(id) + "/summary",
    { json: { style, refresh }, timeout: TIMEOUT_MS.generate, hint: GENERATE_HINT }),
  quiz: (id, body) => request("POST", "/documents/" + encodeURIComponent(id) + "/quiz",
    { json: body, timeout: TIMEOUT_MS.generate, hint: GENERATE_HINT }),
  listConversations: (limit, offset) => request("GET", "/conversations?limit=" + encodeURIComponent(limit) + "&offset=" + encodeURIComponent(offset)),
  getConversation: (id) => request("GET", "/conversations/" + encodeURIComponent(id)),
  deleteConversation: (id) => request("DELETE", "/conversations/" + encodeURIComponent(id)),
};

// ====================================================================== state
const state = {
  docs: [],
  checked: new Set(), // doc_ids ticked in the scope list
  knownReady: new Set(), // ready docs already offered once (new ones are ticked automatically)
  processingSince: new Map(), // doc_id -> ms timestamp when we first saw it processing
  pollTimer: null,
  pollStopped: false,
  confirmDoc: null,
  conversationId: null,
  asking: false,
  busy: new Set(), // ids of summary/quiz buttons with a request in flight
  docsSig: "", // signature of what the document list last rendered (skip identical re-renders)
  history: { items: [], total: 0, shown: 0 },
  confirmConv: null,
  quizRun: 0,
};

// ====================================================================== DOM helpers
const $ = (id) => document.getElementById(id);

function el(tag, opts, children) {
  const node = document.createElement(tag);
  const o = opts || {};
  if (o.className) node.className = o.className;
  if (o.text !== undefined && o.text !== null) node.textContent = o.text;
  if (o.attrs) for (const [k, v] of Object.entries(o.attrs)) node.setAttribute(k, v);
  if (children) for (const child of children) if (child) node.appendChild(child);
  return node;
}

function badge(text, tone) {
  return el("span", { className: "badge " + tone, text });
}

function box(text, tone) {
  return el("div", { className: "box " + tone, text, attrs: { role: tone === "danger" ? "alert" : "status" } });
}

function clear(node) {
  while (node.firstChild) node.removeChild(node.firstChild);
}

function pageText(page) {
  return page !== null && page !== undefined ? "trang " + page : "không có số trang";
}

function showNotice(tone, text) {
  const area = $("notice");
  clear(area);
  area.appendChild(box(text, tone));
}

function clearNotice() {
  clear($("notice"));
}

function docById(id) {
  return state.docs.find((d) => d.doc_id === id) || null;
}

function readyDocs() {
  return state.docs.filter((d) => d.status === "ready");
}

// ====================================================================== renderers: health + documents
function setHealth(ok, degraded) {
  const node = $("health");
  if (!ok) {
    node.textContent = "Mất kết nối";
    node.className = "badge danger";
  } else if (degraded) {
    node.textContent = "Qdrant không kết nối";
    node.className = "badge warning";
  } else {
    node.textContent = "Hoạt động";
    node.className = "badge success";
  }
  $("backend-down").hidden = ok;
}

async function checkHealth() {
  try {
    const { data } = await api.health();
    setHealth(true, data.status !== "ok" || data.qdrant_reachable === false);
    return true;
  } catch (err) {
    setHealth(false, false);
    $("backend-down-text").textContent = err.message;
    return false;
  }
}

function docMeta(doc) {
  const bits = [];
  if (doc.num_pages !== null && doc.num_pages !== undefined) bits.push(doc.num_pages + " trang");
  if (doc.num_chunks !== null && doc.num_chunks !== undefined) bits.push(doc.num_chunks + " đoạn");
  return bits.join(" · ");
}

function renderDocuments() {
  const list = $("doc-list");
  clear(list);
  $("doc-empty").hidden = state.docs.length > 0;
  for (const doc of state.docs) {
    const [label, tone] = STATUS_BADGE[doc.status] || [String(doc.status), "neutral"];
    const item = el("li", { className: "doc-item" });
    item.appendChild(el("div", { className: "doc-name", text: doc.filename }));
    const meta = el("div", { className: "doc-meta" }, [badge(label, tone)]);
    const metaText = docMeta(doc);
    if (metaText) meta.appendChild(el("span", { className: "caption", text: metaText }));
    item.appendChild(meta);
    if (doc.status === "failed") {
      item.appendChild(el("div", { className: "caption", text: (doc.error_message || "Xử lý thất bại.") + " Hãy tải lại tệp để thử lại." }));
    }
    if (doc.status === "processing") {
      const since = state.processingSince.get(doc.doc_id);
      if (since && Date.now() - since > SLOW_AFTER_MS) {
        item.appendChild(box("Vẫn đang xử lý. Embedding của gói miễn phí chậm (khoảng 100 đoạn mỗi phút) nên tài liệu dài có thể mất vài phút. Bạn vẫn hỏi được các tài liệu đã sẵn sàng.", "warning"));
      }
    }
    if (state.confirmDoc === doc.doc_id) {
      const confirm = el("div", { className: "confirm" }, [box("Xóa tài liệu này cùng dữ liệu đã lập chỉ mục?", "warning")]);
      const yes = el("button", { className: "btn danger", text: "Xác nhận xóa", attrs: { type: "button" } });
      yes.addEventListener("click", () => deleteDocument(doc.doc_id));
      const no = el("button", { className: "btn", text: "Hủy", attrs: { type: "button" } });
      no.addEventListener("click", () => { state.confirmDoc = null; renderDocuments(); });
      confirm.appendChild(el("div", { className: "item-actions" }, [yes, no]));
      item.appendChild(confirm);
    } else {
      const del = el("button", { className: "btn", text: "Xóa", attrs: { type: "button" } });
      if (doc.status === "processing") {
        del.disabled = true;
        del.setAttribute("title", "Không xóa được khi tài liệu đang xử lý");
      }
      del.addEventListener("click", () => { state.confirmDoc = doc.doc_id; renderDocuments(); });
      item.appendChild(el("div", { className: "item-actions" }, [del]));
    }
    list.appendChild(item);
  }
  renderPollInfo();
  renderScope();
  renderDocSelects();
}

function renderPollInfo() {
  const processing = state.docs.some((d) => d.status === "processing");
  const info = $("poll-info");
  if (processing && !state.pollStopped) {
    info.textContent = "Đang xử lý; danh sách tự cập nhật.";
  } else if (processing && state.pollStopped) {
    info.textContent = "Đã dừng tự động kiểm tra sau 3 phút. Tài liệu có thể vẫn đang được xử lý ở máy chủ.";
  } else {
    info.textContent = "";
  }
  $("recheck").hidden = !(processing && state.pollStopped);
}

function renderScope() {
  const box_ = $("scope-list");
  clear(box_);
  const ready = readyDocs();
  $("scope-empty").hidden = ready.length > 0;
  ready.forEach((doc, i) => {
    const input = el("input", { attrs: { type: "checkbox", id: "scope-" + i } });
    input.checked = state.checked.has(doc.doc_id);
    input.addEventListener("change", () => {
      if (input.checked) state.checked.add(doc.doc_id); else state.checked.delete(doc.doc_id);
      updateChatAvailability();
    });
    const label = el("label", { attrs: { for: "scope-" + i } }, [input, el("span", { text: doc.filename })]);
    box_.appendChild(label);
  });
  updateChatAvailability();
}

function selectedDocIds() {
  return readyDocs().filter((d) => state.checked.has(d.doc_id)).map((d) => d.doc_id);
}

function disabledReason() {
  if ($("backend-down").hidden === false) return "Không kết nối được máy chủ API";
  if (readyDocs().length === 0) return "Hãy tải lên một tài liệu và đợi trạng thái “Sẵn sàng”";
  if (selectedDocIds().length === 0) return "Hãy chọn ít nhất một tài liệu ở thanh bên";
  return null;
}

function updateChatAvailability() {
  const reason = disabledReason();
  const blocked = Boolean(reason) || state.asking;
  $("question").disabled = Boolean(reason);
  $("send-btn").disabled = blocked;
  $("question").placeholder = reason || "Đặt câu hỏi về tài liệu đã chọn…";
}

function renderDocSelects() {
  for (const id of ["summary-doc", "quiz-doc"]) {
    const select = $(id);
    const previous = select.value;
    clear(select);
    const ready = readyDocs();
    for (const doc of ready) select.appendChild(el("option", { text: doc.filename, attrs: { value: doc.doc_id } }));
    if (ready.some((d) => d.doc_id === previous)) select.value = previous;
    select.disabled = ready.length === 0;
  }
  const none = readyDocs().length === 0;
  for (const id of ["summary-make", "summary-again", "quiz-make", "quiz-new"]) $(id).disabled = none || state.busy.has(id);
}

function syncDocState() {
  const now = Date.now();
  const ids = new Set(state.docs.map((d) => d.doc_id));
  for (const doc of state.docs) {
    if (doc.status === "processing") {
      if (!state.processingSince.has(doc.doc_id)) state.processingSince.set(doc.doc_id, now);
    } else {
      state.processingSince.delete(doc.doc_id);
    }
    if (doc.status === "ready" && !state.knownReady.has(doc.doc_id)) {
      state.knownReady.add(doc.doc_id);
      state.checked.add(doc.doc_id);
    }
  }
  for (const id of [...state.checked]) if (!ids.has(id)) state.checked.delete(id);
  for (const id of [...state.knownReady]) if (!ids.has(id)) state.knownReady.delete(id);
  for (const id of [...state.processingSince.keys()]) if (!ids.has(id)) state.processingSince.delete(id);
  if (state.confirmDoc && !ids.has(state.confirmDoc)) state.confirmDoc = null;
}

async function refreshDocs() {
  try {
    const { data } = await api.listDocuments();
    if (!Array.isArray(data)) throw new ApiError("bad_response", "Máy chủ trả về dữ liệu không đúng định dạng. Hãy thử lại sau.");
    state.docs = data;
    syncDocState();
    // Re-render only when something visible changed, so keyboard focus and open <select> popups survive the 2s poll.
    const now = Date.now();
    const slow = data.filter((d) => d.status === "processing" && now - (state.processingSince.get(d.doc_id) || now) > SLOW_AFTER_MS).map((d) => d.doc_id);
    const sig = JSON.stringify([data, slow]);
    if (sig !== state.docsSig) {
      state.docsSig = sig;
      renderDocuments();
    }
    schedulePoll();
    return true;
  } catch (err) {
    showNotice("danger", "Không cập nhật được danh sách tài liệu: " + err.message);
    if (err.kind === "unreachable") checkHealth();
    schedulePoll();
    return false;
  }
}

function oldestProcessingAge() {
  let oldest = 0;
  for (const [, since] of state.processingSince) oldest = Math.max(oldest, Date.now() - since);
  return oldest;
}

function schedulePoll() {
  clearTimeout(state.pollTimer);
  state.pollTimer = null;
  const processing = state.docs.some((d) => d.status === "processing");
  if (!processing) {
    state.pollStopped = false;
    renderPollInfo();
    return;
  }
  if (oldestProcessingAge() > STOP_AFTER_MS) {
    state.pollStopped = true;
    renderPollInfo();
    return;
  }
  state.pollStopped = false;
  state.pollTimer = setTimeout(refreshDocs, POLL_MS);
}

// ====================================================================== actions: upload + delete
async function uploadFile() {
  const input = $("file-input");
  const file = input.files && input.files[0];
  if (!file) {
    showNotice("warning", "Hãy chọn một tệp PDF, DOCX hoặc TXT trước.");
    return;
  }
  const lower = file.name.toLowerCase();
  if (!ALLOWED_EXT.some((ext) => lower.endsWith(ext))) {
    showNotice("danger", "Định dạng tệp không được hỗ trợ. Chỉ chấp nhận .pdf, .docx và .txt.");
    return;
  }
  if (file.size > MAX_UPLOAD_MB * 1024 * 1024) {
    showNotice("danger", "Tệp «" + file.name + "» vượt quá " + MAX_UPLOAD_MB + " MB, nên chưa được gửi lên.");
    return;
  }
  const button = $("upload-btn");
  button.disabled = true;
  showNotice("info", "Đang gửi tệp lên máy chủ…");
  try {
    const { status } = await api.upload(file);
    if (status === 200) showNotice("info", "«" + file.name + "» đã có sẵn và đã sẵn sàng, không cần tải lại.");
    else showNotice("success", "Đã nhận «" + file.name + "». Hệ thống đang xử lý ở nền; bạn có thể hỏi các tài liệu khác trong lúc chờ.");
    input.value = "";
    state.pollStopped = false;
    // Give every processing document a fresh 180s window, so an old stuck one cannot stop polling for the new upload.
    state.processingSince.clear();
    await refreshDocsKeepNotice();
  } catch (err) {
    showNotice("danger", "Không tải lên được «" + file.name + "»: " + err.message);
    if (err.kind === "unreachable") checkHealth();
  } finally {
    button.disabled = false;
  }
}

async function refreshDocsKeepNotice() {
  const notice = $("notice").firstChild;
  const saved = notice ? notice.cloneNode(true) : null;
  const ok = await refreshDocs();
  if (ok && saved) {
    clearNotice();
    $("notice").appendChild(saved);
  }
}

async function deleteDocument(docId) {
  const doc = docById(docId);
  const name = doc ? doc.filename : docId;
  state.confirmDoc = null;
  try {
    const { data } = await api.deleteDocument(docId);
    state.checked.delete(docId);
    showNotice("success", "Đã xóa «" + name + "» (" + data.deleted_chunks + " đoạn).");
  } catch (err) {
    showNotice("danger", "Không xóa được «" + name + "»: " + err.message);
  }
  await refreshDocsKeepNotice();
}

// ====================================================================== renderers: chat + sources
function sourceNode(source, idPrefix) {
  const wrap = el("div", { className: "source", attrs: { id: idPrefix + source.number } });
  const head = el("div", { className: "source-head" }, [
    el("strong", { text: "[" + source.number + "]" }),
    el("span", { text: source.filename }),
    el("span", { className: "caption", text: pageText(source.page) }),
    el("span", { className: "caption mono", text: "điểm " + Number(source.score).toFixed(3) }),
    source.cited ? badge("Được trích dẫn", "success") : badge("Chỉ truy xuất", "neutral"),
  ]);
  if (source.document_exists === false) head.appendChild(badge("Tài liệu đã bị xóa", "danger"));
  wrap.appendChild(head);
  wrap.appendChild(el("p", { className: "excerpt", text: source.excerpt || "" }));
  return wrap;
}

function highlightSource(details, node) {
  details.open = true;
  for (const other of details.querySelectorAll(".source")) other.classList.remove("highlight");
  node.classList.add("highlight");
  node.scrollIntoView({ block: "nearest", behavior: "smooth" });
  const focusTarget = node.querySelector("strong");
  if (focusTarget) {
    focusTarget.setAttribute("tabindex", "-1");
    focusTarget.focus({ preventScroll: true });
  }
}

// Turns "... [1] ... [2][3]" into text nodes and buttons, built from DOM nodes only.
function answerNode(answer, sourceNodes, details) {
  const out = el("div", { className: "text" });
  const pattern = /\[(\d+)\]/g;
  let last = 0;
  let match;
  while ((match = pattern.exec(answer)) !== null) {
    const number = parseInt(match[1], 10);
    const target = sourceNodes.get(number);
    if (!target) continue; // not a known source: leave it as plain text
    if (match.index > last) out.appendChild(document.createTextNode(answer.slice(last, match.index)));
    const button = el("button", { className: "cite", text: "[" + number + "]", attrs: { type: "button", "aria-label": "Xem nguồn " + number } });
    button.addEventListener("click", () => highlightSource(details, target));
    out.appendChild(button);
    last = match.index + match[0].length;
  }
  if (last < answer.length) out.appendChild(document.createTextNode(answer.slice(last)));
  return out;
}

function latencyText(latency, usedFallback) {
  if (!latency) return "";
  const parts = ["nhúng câu hỏi " + Math.round(latency.embed_ms) + " ms", "tìm kiếm " + Math.round(latency.search_ms) + " ms"];
  parts.push(usedFallback ? "không gọi mô hình" : "mô hình " + (latency.llm_ms / 1000).toFixed(1) + " s");
  return "Thời gian: " + parts.join(" · ") + " · tổng " + (latency.total_ms / 1000).toFixed(1) + " s";
}

function fallbackCaption(bestScore) {
  if (bestScore === null || bestScore === undefined) return "Không tìm thấy đoạn nào trong các tài liệu đã chọn.";
  return "Điểm tương đồng cao nhất: " + Number(bestScore).toFixed(3) + ", thấp hơn ngưỡng của hệ thống, nên mô hình không được gọi (tránh trả lời bịa).";
}

let messageCounter = 0;

function assistantNode(result, saved) {
  messageCounter += 1;
  const idPrefix = "src-" + messageCounter + "-";
  const wrap = el("div", { className: "msg assistant" }, [el("div", { className: "msg-role", text: "Trợ lý" })]);
  if (result.used_fallback) {
    wrap.appendChild(el("div", { className: "box info" }, [
      el("strong", { text: "Không đủ bằng chứng trong tài liệu để trả lời." }),
      el("div", { className: "text", text: result.answer }),
    ]));
    wrap.appendChild(el("p", { className: "caption", text: fallbackCaption(result.best_score) }));
  } else {
    const sources = Array.isArray(result.sources) ? result.sources : [];
    const nodes = new Map();
    const details = el("details", { className: "sources" });
    if (sources.length) {
      details.appendChild(el("summary", { text: "Nguồn (" + sources.length + ")" }));
      if (!sources.some((s) => s.cited)) {
        details.appendChild(el("p", { className: "caption", text: "Câu trả lời không trích dẫn nguồn nào; các đoạn dưới đây chỉ là phần được truy xuất." }));
      }
      for (const source of sources) {
        const node = sourceNode(source, idPrefix);
        nodes.set(source.number, node);
        details.appendChild(node);
      }
    }
    wrap.appendChild(answerNode(result.answer, nodes, details));
    if (result.truncated) wrap.appendChild(box("Câu trả lời có thể đã bị cắt do giới hạn độ dài.", "warning"));
    if (sources.length) wrap.appendChild(details);
  }
  const timing = latencyText(result.latency, result.used_fallback);
  if (timing) wrap.appendChild(el("p", { className: "caption mono", text: timing }));
  if (saved === false) wrap.appendChild(box(SAVE_FAILED_TEXT, "warning"));
  return wrap;
}

function userNode(text) {
  return el("div", { className: "msg user" }, [el("div", { className: "msg-role", text: "Bạn" }), el("div", { className: "text", text })]);
}

function scrollChatToEnd() {
  const log = $("chat-log");
  const last = log.lastElementChild;
  if (last) last.scrollIntoView({ block: "nearest" });
}

function renderEmptyChat() {
  const log = $("chat-log");
  clear(log);
  const hint = box(disabledReason() || "Hãy đặt câu hỏi ở ô bên dưới. Câu trả lời sẽ kèm nguồn trích dẫn [n].", "info");
  hint.classList.add("empty-hint");
  log.appendChild(hint);
}

function warnAsking() {
  showNotice("warning", "Đang chờ câu trả lời. Hãy đợi xong rồi mới đổi cuộc trò chuyện.");
}

function resetConversation() {
  if (state.asking) { warnAsking(); return; }
  state.conversationId = null;
  renderEmptyChat();
  renderHistory();
}

async function sendQuestion(event) {
  event.preventDefault();
  if (state.asking) return;
  const input = $("question");
  const question = input.value.trim();
  if (!question || disabledReason()) return;
  const docIds = selectedDocIds();
  state.asking = true;
  updateChatAvailability();
  const log = $("chat-log");
  if (log.firstChild && log.firstChild.classList.contains("empty-hint")) clear(log);
  log.appendChild(userNode(question));
  const pending = box("Đang tìm trong tài liệu và hỏi mô hình… (tối đa khoảng 60 giây)", "neutral");
  log.appendChild(pending);
  scrollChatToEnd();
  const body = { question, doc_ids: docIds };
  if (state.conversationId) body.conversation_id = state.conversationId;
  try {
    const { data } = await api.chat(body);
    pending.remove();
    log.appendChild(assistantNode(data, data.saved));
    input.value = "";
    if (data.conversation_id) state.conversationId = data.conversation_id;
    loadHistory(true);
  } catch (err) {
    pending.remove();
    if (err.code === "conversation_not_found") {
      state.conversationId = null;
      log.appendChild(box("Cuộc trò chuyện đã lưu không còn tồn tại nên hệ thống bắt đầu một cuộc mới. Hãy gửi lại câu hỏi.", "warning"));
      renderHistory();
    } else {
      log.appendChild(box(err.message, err.kind === "timeout" ? "warning" : "danger"));
    }
    if (err.kind === "unreachable") checkHealth();
  } finally {
    state.asking = false;
    updateChatAvailability();
    scrollChatToEnd();
  }
}

// ====================================================================== renderers: history
function conversationMeta(info) {
  const when = info.updated_at ? new Date(info.updated_at) : null;
  const parts = [];
  if (when && !Number.isNaN(when.getTime())) parts.push(when.toLocaleString("vi-VN"));
  parts.push(info.message_count + " tin nhắn", (info.doc_ids || []).length + " tài liệu");
  if ((info.missing_doc_ids || []).length) parts.push(info.missing_doc_ids.length + " đã bị xóa");
  return parts.join(" · ");
}

function renderHistory() {
  const list = $("history-list");
  clear(list);
  const items = state.history.items;
  $("history-empty").hidden = items.length > 0;
  for (const info of items) {
    const isOpen = info.id === state.conversationId;
    const title = el("div", { className: "doc-name", text: info.title || "(không có tiêu đề)" });
    if (isOpen) title.appendChild(document.createTextNode(" "));
    const item = el("li", { className: "history-item" }, [title]);
    if (isOpen) title.appendChild(badge("Đang mở", "info"));
    item.appendChild(el("div", { className: "caption", text: conversationMeta(info) }));
    if (state.confirmConv === info.id) {
      const yes = el("button", { className: "btn danger", text: "Xác nhận xóa", attrs: { type: "button" } });
      yes.addEventListener("click", () => deleteConversation(info));
      const no = el("button", { className: "btn", text: "Hủy", attrs: { type: "button" } });
      no.addEventListener("click", () => { state.confirmConv = null; renderHistory(); });
      item.appendChild(box("Xóa cuộc trò chuyện này khỏi lịch sử? Không thể hoàn tác.", "warning"));
      item.appendChild(el("div", { className: "item-actions" }, [yes, no]));
    } else {
      const open = el("button", { className: "btn", text: "Mở", attrs: { type: "button" } });
      open.disabled = isOpen;
      open.addEventListener("click", () => openConversation(info.id));
      const del = el("button", { className: "btn", text: "Xóa", attrs: { type: "button" } });
      del.addEventListener("click", () => { state.confirmConv = info.id; renderHistory(); });
      item.appendChild(el("div", { className: "item-actions" }, [open, del]));
    }
    list.appendChild(item);
  }
  $("history-more").hidden = state.history.items.length >= state.history.total;
}

async function loadHistory(reset) {
  const limit = reset ? Math.min(100, Math.max(HISTORY_PAGE, state.history.items.length)) : HISTORY_PAGE; // server allows 1-100
  const offset = reset ? 0 : state.history.items.length;
  try {
    const { data } = await api.listConversations(limit, offset);
    state.history.items = reset ? data.items : state.history.items.concat(data.items);
    state.history.total = data.total;
    renderHistory();
  } catch (err) {
    showNotice("danger", "Không tải được lịch sử: " + err.message);
  }
}

async function openConversation(id) {
  if (state.asking) { warnAsking(); return; }
  try {
    const { data } = await api.getConversation(id);
    const log = $("chat-log");
    clear(log);
    if ((data.missing_doc_ids || []).length) {
      log.appendChild(box(data.missing_doc_ids.length + " tài liệu của cuộc trò chuyện này đã bị xóa khỏi hệ thống. Các nguồn của chúng vẫn được hiển thị (đánh dấu “Tài liệu đã bị xóa”) nhưng không còn tài liệu để đối chiếu.", "warning"));
    }
    for (const m of data.messages) {
      if (m.role === "user") log.appendChild(userNode(m.content));
      else log.appendChild(assistantNode({
        answer: m.content, sources: m.sources || [], used_fallback: Boolean(m.used_fallback),
        best_score: m.best_score, truncated: Boolean(m.truncated), latency: m.latency,
      }, true));
    }
    state.conversationId = data.id;
    // Same as the Streamlit app: scope the next question to this conversation's documents that still exist and are ready.
    const readyIds = new Set(readyDocs().map((d) => d.doc_id));
    state.checked = new Set((data.doc_ids || []).filter((docId) => readyIds.has(docId)));
    renderScope();
    switchTab("chat");
    closeDrawer();
    renderHistory();
    scrollChatToEnd();
  } catch (err) {
    if (err.code === "conversation_not_found") loadHistory(true);
    showNotice("danger", "Không mở được cuộc trò chuyện: " + err.message);
  }
}

async function deleteConversation(info) {
  state.confirmConv = null;
  try {
    const { data } = await api.deleteConversation(info.id);
    showNotice("success", "Đã xóa cuộc trò chuyện «" + (info.title || "(không có tiêu đề)") + "» (" + data.deleted_messages + " tin nhắn).");
    if (state.conversationId === info.id) resetConversation();
  } catch (err) {
    showNotice("danger", "Không xóa được cuộc trò chuyện: " + err.message);
  }
  await loadHistory(true);
}

// ====================================================================== renderers: summary + quiz
function costText(calls, cached, latencyMs) {
  const cost = cached ? "lấy từ kết quả đã lưu (0 lần gọi mô hình)" : calls + " lần gọi mô hình";
  return cost + " · " + (latencyMs / 1000).toFixed(1) + " s";
}

function setBusy(buttons, busy) {
  for (const id of buttons) {
    if (busy) state.busy.add(id); else state.busy.delete(id);
    $(id).disabled = busy || readyDocs().length === 0;
  }
}

async function makeSummary(refresh) {
  const docId = $("summary-doc").value;
  if (!docId) return;
  const style = document.querySelector('input[name="summary-style"]:checked').value;
  const out = $("summary-out");
  clear(out);
  out.appendChild(box("Đang tạo… có thể mất vài chục giây (tối đa khoảng 2 phút). Kết quả được lưu để lần sau không tốn thêm lượt gọi.", "neutral"));
  setBusy(["summary-make", "summary-again"], true);
  try {
    const { data } = await api.summary(docId, style, refresh);
    clear(out);
    if (data.truncated && data.warning) out.appendChild(box(data.warning, "warning"));
    out.appendChild(el("div", { className: "msg" }, [el("div", { className: "text", text: data.summary })]));
    out.appendChild(el("p", { className: "caption", text: "Dùng " + data.num_chunks_used + "/" + data.num_chunks_total + " đoạn · " + costText(data.llm_calls, data.cached, data.latency_ms) }));
  } catch (err) {
    clear(out);
    out.appendChild(box(err.message, err.kind === "timeout" ? "warning" : "danger"));
  } finally {
    setBusy(["summary-make", "summary-again"], false);
  }
}

function optionLabel(index, text) {
  return OPTION_LETTERS[index] + ". " + text;
}

function intOrNull(id) {
  const raw = $(id).value.trim();
  if (!raw) return null;
  const n = parseInt(raw, 10);
  return Number.isNaN(n) || n < 1 ? null : n;
}

async function makeQuiz(refresh) {
  const docId = $("quiz-doc").value;
  if (!docId) return;
  const count = Math.min(10, Math.max(1, parseInt($("quiz-count").value, 10) || 5));
  const body = { num_questions: count, verify: $("quiz-verify").checked, refresh };
  const topic = $("quiz-topic").value.trim();
  if (topic) body.topic = topic;
  const from = intOrNull("quiz-from");
  const to = intOrNull("quiz-to");
  if (from !== null) body.page_from = from;
  if (to !== null) body.page_to = to;
  if (from !== null && to !== null && from > to) {
    const out = $("quiz-out");
    clear(out);
    out.appendChild(box("“Từ trang” không được lớn hơn “Đến trang”.", "danger"));
    return;
  }
  const out = $("quiz-out");
  clear(out);
  out.appendChild(box("Đang tạo… có thể mất vài chục giây (tối đa khoảng 2 phút). Kết quả được lưu để lần sau không tốn thêm lượt gọi.", "neutral"));
  setBusy(["quiz-make", "quiz-new"], true);
  try {
    const { data } = await api.quiz(docId, body);
    renderQuiz(out, data);
  } catch (err) {
    clear(out);
    out.appendChild(box(err.message, err.kind === "timeout" ? "warning" : "danger"));
  } finally {
    setBusy(["quiz-make", "quiz-new"], false);
  }
}

function renderQuiz(out, quiz) {
  clear(out);
  state.quizRun += 1;
  const run = state.quizRun;
  const questions = quiz.questions || [];
  let checkedLabel = "không chạy bước xác minh";
  if (quiz.verify_requested) checkedLabel = quiz.verified ? "đã xác minh" : "chưa xác minh được";
  out.appendChild(el("p", { className: "caption", text: questions.length + "/" + quiz.requested + " câu · " + checkedLabel + " · " + costText(quiz.llm_calls, quiz.cached, quiz.latency_ms) }));
  if (quiz.message) out.appendChild(box(quiz.message, "info"));
  for (const w of quiz.warnings || []) out.appendChild(box(w, "warning"));
  const dropped = Object.entries(quiz.dropped || {}).filter(([, n]) => n > 0).sort((a, b) => b[1] - a[1]);
  if (dropped.length) {
    out.appendChild(el("p", { className: "caption", text: "Các câu đã bị loại: " + dropped.map(([r, n]) => n + " câu " + (DROP_REASONS[r] || r)).join("; ") }));
  }
  for (const d of (quiz.verifier_decisions || []).filter((x) => x.kept === false)) {
    out.appendChild(el("p", { className: "caption", text: "Bị loại: " + d.question + " — " + d.reason }));
  }
  if (!questions.length) return;
  const form = el("form", { attrs: { novalidate: "novalidate" } });
  const cards = [];
  questions.forEach((q, qi) => {
    const fieldset = el("fieldset");
    fieldset.appendChild(el("legend", { text: "Câu " + q.id + ". " + q.question }));
    const inputs = q.options.map((option, oi) => {
      const input = el("input", { attrs: { type: "radio", name: "quiz-" + run + "-" + qi, value: String(oi) } });
      fieldset.appendChild(el("label", {}, [input, el("span", { text: optionLabel(oi, option) })]));
      return input;
    });
    const result = el("div", { className: "quiz-result" });
    const card = el("div", { className: "quiz-q" }, [fieldset, result]);
    cards.push({ q, inputs, result });
    form.appendChild(card);
  });
  const score = el("h3");
  const check = el("button", { className: "btn primary", text: "Kiểm tra", attrs: { type: "submit" } });
  const retry = el("button", { className: "btn", text: "Làm lại bộ câu hỏi này", attrs: { type: "button" } });
  retry.hidden = true;
  form.appendChild(el("div", { className: "row" }, [check, retry]));
  form.appendChild(score);
  form.addEventListener("submit", (event) => {
    event.preventDefault();
    let correct = 0;
    for (const c of cards) {
      const chosen = c.inputs.findIndex((i) => i.checked);
      const key = c.q.answer_index; // known to the page only now: shown after the user pressed "Kiểm tra"
      clear(c.result);
      let line;
      if (chosen === -1) line = box("Chưa trả lời. Đáp án đúng là " + OPTION_LETTERS[key] + ".", "warning");
      else if (chosen === key) { correct += 1; line = box("Đúng (" + OPTION_LETTERS[chosen] + ").", "success"); }
      else line = box("Sai: bạn chọn " + OPTION_LETTERS[chosen] + ", đáp án đúng là " + OPTION_LETTERS[key] + ".", "danger");
      c.result.appendChild(line);
      c.result.appendChild(el("div", {}, [el("strong", { text: "Đáp án đúng: " }), el("span", { text: optionLabel(key, c.q.options[key]) })]));
      c.result.appendChild(el("div", {}, [el("strong", { text: "Giải thích: " }), el("span", { text: c.q.explanation })]));
      const src = c.q.source;
      c.result.appendChild(el("div", { className: "caption", text: "Nguồn: " + src.filename + ", " + pageText(src.page) + " · " + (c.q.verified === true ? "Đã xác minh" : "Chưa xác minh") }));
      c.result.appendChild(el("blockquote", { className: "excerpt", text: src.quote }));
      for (const i of c.inputs) i.disabled = true;
    }
    score.textContent = "Điểm: " + correct + "/" + cards.length;
    check.disabled = true;
    retry.hidden = false;
  });
  retry.addEventListener("click", () => renderQuiz(out, quiz));
  out.appendChild(form);
}

// ====================================================================== tabs + drawer
function switchTab(name) {
  for (const tab of document.querySelectorAll(".tab")) {
    const active = tab.dataset.tab === name;
    tab.classList.toggle("active", active);
    tab.setAttribute("aria-selected", active ? "true" : "false");
    $("panel-" + tab.dataset.tab).hidden = !active;
  }
}

function openDrawer() {
  document.body.classList.add("drawer-open");
  $("backdrop").hidden = false;
  $("close-drawer").focus();
}

function closeDrawer() {
  const wasOpen = document.body.classList.contains("drawer-open");
  document.body.classList.remove("drawer-open");
  $("backdrop").hidden = true;
  if (wasOpen) $("open-drawer").focus();
}

// ====================================================================== wiring + init
function wire() {
  $("upload-btn").addEventListener("click", uploadFile);
  $("chat-form").addEventListener("submit", sendQuestion);
  $("question").addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
      event.preventDefault();
      $("chat-form").requestSubmit();
    }
  });
  $("new-chat").addEventListener("click", resetConversation);
  $("history-more").addEventListener("click", () => loadHistory(false));
  $("summary-make").addEventListener("click", () => makeSummary(false));
  $("summary-again").addEventListener("click", () => makeSummary(true));
  $("quiz-make").addEventListener("click", () => makeQuiz(false));
  $("quiz-new").addEventListener("click", () => makeQuiz(true));
  $("retry-health").addEventListener("click", async () => {
    if (await checkHealth()) { refreshDocs(); loadHistory(true); }
  });
  $("recheck").addEventListener("click", () => {
    for (const id of state.processingSince.keys()) state.processingSince.set(id, Date.now());
    state.pollStopped = false;
    refreshDocs();
  });
  for (const tab of document.querySelectorAll(".tab")) tab.addEventListener("click", () => switchTab(tab.dataset.tab));
  $("open-drawer").addEventListener("click", openDrawer);
  $("close-drawer").addEventListener("click", closeDrawer);
  $("backdrop").addEventListener("click", closeDrawer);
  document.addEventListener("keydown", (event) => { if (event.key === "Escape") closeDrawer(); });
}

async function init() {
  wire();
  renderEmptyChat();
  renderDocSelects();
  const ok = await checkHealth();
  if (ok) {
    await refreshDocs();
    await loadHistory(true);
  }
  updateChatAvailability();
  renderEmptyChat();
}

if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", init);
else init();
