import { useCallback, useEffect, useRef, useState } from "react";
import MarkdownMessage from "./MarkdownMessage.jsx";
import {
  createConversation,
  getConversations,
  getMessages,
  logout,
  streamAgentMessage,
  updateConversation,
  uploadDocument,
} from "../api.js";
import { auth, signOut } from "../firebase.js";
import {
  PanelLeft,
  SquarePen,
  Plus,
  MessageSquare,
  Zap,
  Code,
  FileText,
  Monitor,
  Image as ImageIcon,
  Globe,
  Paperclip,
  Mic,
  Coins,
  LogOut,
  Send,
  Edit3,
} from "./Icons.jsx";

const AGENT_MODES = [
  { id: "auto", label: "Auto", Icon: Zap },
  { id: "chat", label: "Chat", Icon: MessageSquare },
  { id: "coding", label: "Coding", Icon: Code },
  { id: "rag", label: "RAG", Icon: FileText },
  { id: "pdf", label: "PDF", Icon: FileText },
  { id: "ppt", label: "PPT", Icon: Monitor },
  { id: "image", label: "Image", Icon: ImageIcon },
  { id: "search", label: "Search", Icon: Globe },
];

const AGENT_META = {
  chat: { label: "Chat Agent", icon: "💬" },
  coding: { label: "Coding Agent", icon: "💻" },
  pdf: { label: "PDF Generator", icon: "📄" },
  ppt: { label: "PPT Generator", icon: "📊" },
  image: { label: "Image Generator", icon: "🎨" },
  search: { label: "Search Agent", icon: "🔍" },
  rag: { label: "RAG Agent", icon: "📚" },
};

const PROMPT_SUGGESTIONS = [
  "Write a Netflix clone",
  "Explain Redis",
  "Build a dashboard",
];

export default function Chat({ firebaseUser }) {
  const [conversations, setConversations] = useState([]);
  const [activeId, setActiveId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState("");
  const [agent, setAgent] = useState("auto");
  const [sending, setSending] = useState(false);
  const [error, setError] = useState("");
  const [attachedFile, setAttachedFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const bottomRef = useRef(null);
  const fileInputRef = useRef(null);

  const active = conversations.find((c) => c.id === activeId) || null;

  const refreshConversations = useCallback(async () => {
    const list = await getConversations();
    setConversations(list);
    return list;
  }, []);

  useEffect(() => {
    (async () => {
      try {
        const list = await refreshConversations();
        if (list.length === 0) {
          const created = await createConversation();
          setConversations([created]);
          setActiveId(created.id);
        } else {
          setActiveId(list[0].id);
        }
      } catch (err) {
        setError(err.message);
      }
    })();
  }, [refreshConversations]);

  useEffect(() => {
    if (!activeId) return;
    (async () => {
      try {
        const msgs = await getMessages(activeId);
        setMessages(msgs);
      } catch (err) {
        setError(err.message);
      }
    })();
  }, [activeId]);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages]);

  async function handleNewChat() {
    try {
      const created = await createConversation();
      setConversations((prev) => [created, ...prev]);
      setActiveId(created.id);
      setMessages([]);
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleRename(conversation) {
    const title = window.prompt("New title", conversation.title);
    if (!title || title.trim() === "") return;
    try {
      const updated = await updateConversation(conversation.id, title.trim());
      setConversations((prev) =>
        prev.map((c) => (c.id === updated.id ? updated : c))
      );
    } catch (err) {
      setError(err.message);
    }
  }

  async function handleFileChange(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    if (!file.name.toLowerCase().endsWith(".pdf")) {
      setError("Only PDF files are supported for document indexing.");
      return;
    }

    let targetConvId = activeId;
    if (!targetConvId) {
      try {
        const created = await createConversation();
        setConversations([created]);
        setActiveId(created.id);
        targetConvId = created.id;
      } catch (err) {
        setError(err.message);
        return;
      }
    }

    setUploading(true);
    setError("");

    try {
      const res = await uploadDocument(file, targetConvId);
      setAttachedFile({ name: file.name, chunks: res.chunks });
      setAgent("rag");
    } catch (err) {
      setError(err.message || "Failed to upload and index document.");
    } finally {
      setUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = "";
    }
  }

  async function handleSend(customPrompt) {

    const prompt = (typeof customPrompt === "string" ? customPrompt : input).trim();
    if (!prompt || sending) return;

    let targetConvId = activeId;
    if (!targetConvId) {
      try {
        const created = await createConversation();
        setConversations([created]);
        setActiveId(created.id);
        targetConvId = created.id;
      } catch (err) {
        setError(err.message);
        return;
      }
    }

    const userMsg = {
      id: `temp-${Date.now()}`,
      role: "user",
      content: prompt,
      images: [],
    };

    const assistantMsgId = `temp-${Date.now() + 1}`;
    const assistantMsg = {
      id: assistantMsgId,
      role: "assistant",
      content: "",
      images: [],
      agentUsed: agent === "auto" ? "Routing..." : agent,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInput("");
    setSending(true);
    setError("");

    try {
      await streamAgentMessage(targetConvId, prompt, agent, {
        onAgentSelect: (resolvedAgent) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, agentUsed: resolvedAgent }
                : m
            )
          );
        },
        onToken: (token) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId
                ? { ...m, content: m.content + token }
                : m
            )
          );
        },
        onImages: (imgs) => {
          setMessages((prev) =>
            prev.map((m) =>
              m.id === assistantMsgId ? { ...m, images: imgs } : m
            )
          );
        },
      });
      await refreshConversations();
    } catch (err) {
      setError(err.message);
    } finally {
      setSending(false);
    }
  }


  async function handleLogout() {
    try {
      await logout();
    } catch {
      /* ignore server errors on logout */
    }
    await signOut(auth);
  }

  const userName =
    firebaseUser?.displayName ||
    (firebaseUser?.email ? firebaseUser.email.split("@")[0] : "Virtual Code");
  const avatarLetter = userName[0]?.toUpperCase() || "V";

  return (
    <div className="chat-layout">
      {/* Sidebar */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <div className="brand-group">
            <button className="icon-btn" title="Toggle Sidebar">
              <PanelLeft size={18} />
            </button>
            <span className="brand-name">CortexAI</span>
            <span className="badge-free">free</span>
          </div>
          <button
            className="icon-btn"
            title="New Chat"
            onClick={handleNewChat}
          >
            <SquarePen size={18} />
          </button>
        </div>

        <div className="sidebar-action">
          <button className="btn-new-chat" onClick={handleNewChat}>
            <Plus size={18} />
            <span>New Chat</span>
          </button>
        </div>

        <div className="sidebar-section-title">
          {conversations.length === 0
            ? "NO RECENT CONVERSATIONS"
            : "RECENT CONVERSATIONS"}
        </div>

        <div className="conv-list">
          {conversations.map((c) => (
            <div
              key={c.id}
              className={`conv-item ${c.id === activeId ? "active" : ""}`}
              onClick={() => setActiveId(c.id)}
            >
              <MessageSquare size={15} className="conv-icon" />
              <span className="conv-title">{c.title}</span>
              <button
                className="conv-rename"
                title="Rename"
                onClick={(e) => {
                  e.stopPropagation();
                  handleRename(c);
                }}
              >
                <Edit3 size={13} />
              </button>
            </div>
          ))}
        </div>

        <div className="sidebar-footer">
          <div className="user-profile">
            <div className="avatar-wrapper">
              {firebaseUser?.photoURL ? (
                <img
                  src={firebaseUser.photoURL}
                  alt={userName}
                  className="avatar-img"
                />
              ) : (
                <div className="avatar">{avatarLetter}</div>
              )}
              <span className="online-indicator" />
            </div>
            <div className="user-info">
              <div className="user-name">{userName}</div>
              <div className="user-plan">Free Plan</div>
            </div>
          </div>

          <div className="footer-actions">
            <button className="icon-btn coin-btn" title="Coins / Credits">
              <Coins size={18} />
            </button>
            <button
              className="icon-btn logout-btn"
              title="Log out"
              onClick={handleLogout}
            >
              <LogOut size={18} />
            </button>
          </div>
        </div>
      </aside>

      {/* Main Container */}
      <main className="main">
        <header className="chat-topbar">
          <div className="topbar-left">
            <div className="message-counter-pill">
              <MessageSquare size={15} />
              <span>{messages.length} Messages</span>
            </div>
          </div>
          <div className="topbar-right">
            <a
              href="https://logfire.pydantic.dev/"
              target="_blank"
              rel="noopener noreferrer"
              className="logfire-topbar-btn"
              title="View live Logfire observability traces, agent spans, and execution steps"
            >
              <span>🔥 View Logfire Traces</span>
            </a>
          </div>
        </header>


        <div className="workspace">
          {messages.length === 0 ? (
            <div className="hero-section">
              <h1 className="hero-title">CortexAI</h1>
              <h2 className="hero-subtitle">How can I help you?</h2>
              <p className="hero-description">
                Ask me anything — code, ideas, explanations, or just a quick question.
              </p>

              <div className="suggestion-pills">
                {PROMPT_SUGGESTIONS.map((prompt, i) => (
                  <button
                    key={i}
                    className="suggestion-pill"
                    onClick={() => handleSend(prompt)}
                  >
                    {prompt}
                  </button>
                ))}
              </div>
            </div>
          ) : (
            <div className="messages">
              {messages.map((m) => {
                const agentInfo = m.role === "assistant" && m.agentUsed ? AGENT_META[m.agentUsed.toLowerCase()] : null;
                const isRouting = m.role === "assistant" && m.agentUsed === "Routing...";
                return (
                  <div key={m.id} className={`message ${m.role}`}>
                    {m.role === "assistant" && (m.agentUsed || isRouting) && (
                      <div className="agent-badge-pill">
                        <span className="pulse-dot" />
                        <span className="agent-badge-text">
                          {isRouting
                            ? "⚡ Routing request..."
                            : `${agentInfo?.icon || "🤖"} ${agentInfo?.label || m.agentUsed}`}
                        </span>
                      </div>
                    )}
                    <div className="bubble">
                      <div className="bubble-text">
                        {m.role === "assistant" ? (
                          <MarkdownMessage content={m.content} />
                        ) : (
                          m.content
                        )}
                      </div>
                      {m.images?.length > 0 && (
                        <div className="image-grid">
                          {m.images.map((url, i) => (
                            <a key={i} href={url} target="_blank" rel="noreferrer">
                              <img src={url} alt={`result ${i + 1}`} />
                            </a>
                          ))}
                        </div>
                      )}
                    </div>
                  </div>
                );
              })}
              <div ref={bottomRef} />
            </div>

          )}
        </div>

        {error && <div className="banner error">{error}</div>}

        {/* Floating Composer Card */}
        <div className="composer-container">
          <div className="composer-card">
            {/* Agent / Tool Selection Bar */}
            <div className="agent-selector-row">
              {AGENT_MODES.map((mode) => {
                const ModeIcon = mode.Icon;
                const isActive = agent === mode.id;
                return (
                  <button
                    key={mode.id}
                    type="button"
                    className={`agent-tab ${isActive ? "active" : ""}`}
                    onClick={() => setAgent(mode.id)}
                  >
                    <ModeIcon size={15} />
                    <span>{mode.label}</span>
                  </button>
                );
              })}
            </div>

            {/* Hidden File Input for PDF upload */}
            <input
              type="file"
              ref={fileInputRef}
              onChange={handleFileChange}
              accept=".pdf"
              style={{ display: "none" }}
            />

            {/* Attached File Indicator Badge */}
            {attachedFile && (
              <div className="attached-file-badge">
                <FileText size={14} className="file-badge-icon" />
                <span className="file-badge-name">
                  {attachedFile.name} ({attachedFile.chunks} chunks indexed into RAG)
                </span>
                <button
                  type="button"
                  className="file-badge-remove"
                  onClick={() => setAttachedFile(null)}
                  title="Remove document context"
                >
                  ×
                </button>
              </div>
            )}

            {uploading && (
              <div className="attached-file-badge uploading">
                <span className="pulse-dot" />
                <span>Uploading and indexing PDF into vector database...</span>
              </div>
            )}

            {/* Input Form */}
            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSend();
              }}
            >
              <textarea
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    handleSend();
                  }
                }}
                placeholder={
                  attachedFile
                    ? `Ask anything about ${attachedFile.name}...`
                    : "Ask CortexAI..."
                }
                rows={1}
                disabled={sending || uploading}
                className="composer-input"
              />

              <div className="composer-bottom-bar">
                <div className="composer-tools">
                  <button
                    type="button"
                    className={`tool-btn ${attachedFile ? "has-file" : ""}`}
                    title="Attach PDF Document for RAG Search"
                    onClick={() => fileInputRef.current?.click()}
                    disabled={uploading || sending}
                  >
                    <Paperclip size={18} />
                  </button>
                  <button
                    type="button"
                    className="tool-btn"
                    title="Voice input"
                  >
                    <Mic size={18} />
                  </button>
                </div>

                <button
                  type="submit"
                  className={`send-btn ${input.trim() ? "active" : ""}`}
                  disabled={sending || uploading || !input.trim()}
                  title="Send message"
                >
                  <Send size={15} />
                </button>
              </div>
            </form>
          </div>


          <div className="disclaimer-text">
            CortexAI can make mistakes. Verify important info.
          </div>
        </div>
      </main>
    </div>
  );
}
