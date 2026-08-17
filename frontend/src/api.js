import axios from "axios";

const api = axios.create({
  baseURL: import.meta.env.VITE_SERVER_URL || "http://localhost:8000",
  withCredentials: true,
});

api.interceptors.response.use(
  (response) => response.data,
  (error) => {
    const detail =
      error?.response?.data?.detail || error.message || "Request failed";
    throw new Error(typeof detail === "string" ? detail : JSON.stringify(detail));
  }
);

export const loginWithFirebaseToken = (token) => api.post("/api/v1/auth/login", { token });
export const logout = () => api.post("/api/v1/auth/logout");
export const getMe = () => api.get("/api/v1/me");

export const createConversation = () => api.post("/api/v1/chat/create_conversation");
export const getConversations = () => api.get("/api/v1/chat/get_conversations");
export const updateConversation = (conversationId, title) =>
  api.post("/api/v1/chat/update_conversation", { conversation_id: conversationId, title });
export const getMessages = (conversationId) => api.get(`/api/v1/chat/get_messages/${conversationId}`);
export const saveMessage = (conversationId, role, content, images = []) =>
  api.post("/api/v1/chat/save_message", { conversation_id: conversationId, role, content, images });

export const sendAgentMessage = (conversationId, prompt, agent = "auto") =>
  api.post("/api/v1/agent/chat", { prompt, conversation_id: conversationId, agent });

export const uploadDocument = (file, conversationId) => {
  const formData = new FormData();
  formData.append("file", file);
  formData.append("conversation_id", conversationId);
  return api.post("/api/v1/documents/upload", formData, {
    headers: { "Content-Type": "multipart/form-data" },
  });
};


export async function streamAgentMessage(conversationId, prompt, agent = "auto", { onAgentSelect, onPlan, onToken, onImages, onComplete }) {
  const baseURL = import.meta.env.VITE_SERVER_URL || "http://localhost:8000";
  const response = await fetch(`${baseURL}/api/v1/agent/chat/stream`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    credentials: "include",
    body: JSON.stringify({ prompt, conversation_id: conversationId, agent }),
  });

  if (!response.ok) {
    const errText = await response.text();
    throw new Error(errText || `Streaming failed with status ${response.status}`);
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder("utf-8");
  let buffer = "";

  while (true) {
    const { done, value } = await reader.read();
    if (done) break;
    buffer += decoder.decode(value, { stream: true });

    const lines = buffer.split("\n\n");
    buffer = lines.pop() || "";

    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed.startsWith("data: ")) continue;
      const dataStr = trimmed.slice(6).trim();
      if (dataStr === "[DONE]") {
        onComplete?.();
        return;
      }

      try {
        const parsed = JSON.parse(dataStr);
        if (parsed.agent && onAgentSelect) {
          onAgentSelect(parsed.agent);
        }
        if (parsed.plan && onPlan) {
          onPlan(parsed.plan);
        }
        if (parsed.token && onToken) {
          onToken(parsed.token);
        }
        if (parsed.images && onImages) {
          onImages(parsed.images);
        }
      } catch (err) {
        console.error("Error parsing SSE JSON chunk:", err, dataStr);
      }
    }
  }
}
