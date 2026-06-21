"use client";

import type { ReactNode, SVGProps } from "react";
import { useRef, useState, useEffect, useCallback } from "react";
import Image from "next/image";

const API = process.env.NEXT_PUBLIC_API_URL ?? "http://localhost:8000";

type Citation = { title: string; page_number: number; chunk_id: string };
type Confidence = "low" | "medium" | "high";
type Message = {
  id: string; role: "user" | "assistant"; text: string;
  citations?: Citation[]; confidence?: Confidence;
  traceId?: string; latency?: number; feedback?: "up" | "down";
};
type UploadJob = {
  jobId: string; files: string[];
  status: "pending" | "processing" | "done" | "failed"; step?: string;
};
type DocMeta = { document_id: string; title: string; file_name: string; pages: number; scanned: boolean };
type ResearchStage = { label: string; description: string };

type IconProps = SVGProps<SVGSVGElement> & { size?: number };

function IconBase({ size = 16, style, children, ...props }: IconProps & { children: ReactNode }) {
  return (
    <svg
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="2"
      strokeLinecap="round"
      strokeLinejoin="round"
      width={size}
      height={size}
      style={style}
      aria-hidden="true"
      {...props}
    >
      {children}
    </svg>
  );
}

function Send(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M22 2 11 13" />
      <path d="m22 2-7 20-4-9-9-4Z" />
    </IconBase>
  );
}

function Upload(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M12 17V3" />
      <path d="m7 8 5-5 5 5" />
      <path d="M5 21h14" />
    </IconBase>
  );
}

function X(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M18 6 6 18" />
      <path d="m6 6 12 12" />
    </IconBase>
  );
}

function FileText(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z" />
      <path d="M14 2v6h6" />
      <path d="M16 13H8" />
      <path d="M16 17H8" />
      <path d="M10 9H8" />
    </IconBase>
  );
}

function Loader2(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M21 12a9 9 0 1 1-6.22-8.56" />
    </IconBase>
  );
}

function User(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M19 21a7 7 0 0 0-14 0" />
      <circle cx="12" cy="8" r="4" />
    </IconBase>
  );
}

function ThumbsUp(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M7 10v11" />
      <path d="M15 5.9 14 10h5.4a2 2 0 0 1 2 2.4l-1.1 5.6A2 2 0 0 1 18.3 20H7a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.8a2 2 0 0 0 1.8-1.1L14 3a1.5 1.5 0 0 1 1 2.9Z" />
    </IconBase>
  );
}

function ThumbsDown(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M17 14V3" />
      <path d="M9 18.1 10 14H4.6a2 2 0 0 1-2-2.4l1.1-5.6A2 2 0 0 1 5.7 4H17a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.8a2 2 0 0 0-1.8 1.1L10 21a1.5 1.5 0 0 1-1-2.9Z" />
    </IconBase>
  );
}

function CheckCircle(props: IconProps) {
  return (
    <IconBase {...props}>
      <circle cx="12" cy="12" r="10" />
      <path d="m9 12 2 2 4-4" />
    </IconBase>
  );
}

function AlertCircle(props: IconProps) {
  return (
    <IconBase {...props}>
      <circle cx="12" cy="12" r="10" />
      <path d="M12 8v4" />
      <path d="M12 16h.01" />
    </IconBase>
  );
}

function Sparkles(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m12 3 1.9 4.9L19 10l-5.1 2.1L12 17l-1.9-4.9L5 10l5.1-2.1Z" />
      <path d="M5 3v4" />
      <path d="M3 5h4" />
      <path d="M19 17v4" />
      <path d="M17 19h4" />
    </IconBase>
  );
}

function BookOpen(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M2 6.5A2.5 2.5 0 0 1 4.5 4H10v16H4.5A2.5 2.5 0 0 0 2 22z" />
      <path d="M22 6.5A2.5 2.5 0 0 0 19.5 4H14v16h5.5A2.5 2.5 0 0 1 22 22z" />
    </IconBase>
  );
}

function ChevronDown(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m6 9 6 6 6-6" />
    </IconBase>
  );
}

function ChevronUp(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="m18 15-6-6-6 6" />
    </IconBase>
  );
}

function Microscope(props: IconProps) {
  return (
    <IconBase {...props}>
      <path d="M6 18h8" />
      <path d="M3 22h18" />
      <path d="M14 22a7 7 0 0 0 0-14h-1" />
      <path d="M9 14h2" />
      <path d="M9 6h4" />
      <path d="M10 6V3" />
      <path d="m14 6-3 8" />
      <path d="M8 10 5 8" />
    </IconBase>
  );
}

function ConfidenceBadge({ level }: { level: Confidence }) {
  const map: Record<Confidence, { cls: string; dot: string; label: string }> = {
    high: { cls: "bg-emerald-500/15 text-emerald-400 border-emerald-500/25", dot: "bg-emerald-400 shadow-[0_0_6px_#34d399]", label: "High confidence" },
    medium: { cls: "bg-amber-500/15  text-amber-400  border-amber-500/25", dot: "bg-amber-400  shadow-[0_0_6px_#fbbf24]", label: "Medium confidence" },
    low: { cls: "bg-red-500/15    text-red-400    border-red-500/25", dot: "bg-red-400    shadow-[0_0_6px_#f87171]", label: "Low confidence" },
  };
  const { cls, dot, label } = map[level];
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${cls}`}>
      <span className={`h-1.5 w-1.5 rounded-full ${dot}`} />
      {label}
    </span>
  );
}

function CitationsPanel({ citations }: { citations: Citation[] }) {
  const [open, setOpen] = useState(false);
  if (!citations.length) return null;
  return (
    <div className="mt-3 overflow-hidden rounded-xl border border-violet-500/20 bg-violet-500/5">
      <button
        onClick={() => setOpen((v) => !v)}
        className="flex w-full items-center justify-between px-4 py-2.5 text-xs font-semibold text-violet-400 transition-colors hover:bg-violet-500/10"
      >
        <span className="flex items-center gap-2"><BookOpen size={12} />{citations.length} Source{citations.length > 1 ? "s" : ""}</span>
        {open ? <ChevronUp size={12} /> : <ChevronDown size={12} />}
      </button>
      {open && (
        <div className="divide-y divide-violet-500/10 border-t border-violet-500/20">
          {citations.map((c, i) => (
            <div key={i} className="flex items-start gap-3 px-4 py-3">
              <span className="mt-0.5 flex h-5 w-5 shrink-0 items-center justify-center rounded-full bg-violet-500/20 text-xs font-bold text-violet-400">{i + 1}</span>
              <div>
                <p className="text-xs font-medium leading-snug text-slate-200">{c.title}</p>
                <p className="mt-0.5 text-xs text-slate-500">Page {c.page_number}</p>
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}

function Modal({ title, subtitle, accentFrom, accentTo, children }: {
  title: string; subtitle: string;
  accentFrom: string; accentTo: string; children: ReactNode;
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 px-4 backdrop-blur-md">
      <div className="w-full max-w-md overflow-hidden rounded-2xl border border-white/10 bg-[#0d1a2e] shadow-2xl shadow-black/60">
        <div className={`bg-gradient-to-r ${accentFrom} ${accentTo} px-6 py-4`}>
          <h2 className="text-sm font-semibold text-white">{title}</h2>
          <p className="mt-0.5 text-xs text-white/70">{subtitle}</p>
        </div>
        <div className="px-6 py-5">{children}</div>
      </div>
    </div>
  );
}

function AssistantLoader({ statusText }: { statusText: string }) {
  const stages: ResearchStage[] = [
    { label: "Retrieve", description: "Scanning the selected paper set" },
    { label: "Compare", description: "Linking the most relevant passages" },
    { label: "Compose", description: "Drafting a cited response" },
  ];

  const normalizedStatus = statusText.toLowerCase();
  const activeStage = normalizedStatus.includes("retriev")
    ? 0
    : normalizedStatus.includes("chunk") || normalizedStatus.includes("match") || normalizedStatus.includes("source")
      ? 1
      : 2;

  return (
    <div className="flex items-start gap-3">
      <div
        className="relative flex h-9 w-9 shrink-0 items-center justify-center rounded-xl"
        style={{ background: "linear-gradient(135deg,#0e7490,#4f46e5)", boxShadow: "0 0 16px rgba(6,182,212,0.25)" }}
      >
        <span className="absolute inset-0 rounded-xl border border-cyan-300/25 animate-[loaderPulse_2s_ease-in-out_infinite]" />
        <Image src="/flamingo.png" alt="Flamingo" width={20} height={20} className="relative z-10 brightness-0 invert" />
      </div>
      <div
        className="min-w-[280px] max-w-[82%] overflow-hidden rounded-2xl rounded-tl-sm"
        style={{
          background: "linear-gradient(180deg, rgba(10,22,40,0.98) 0%, rgba(8,18,33,0.98) 100%)",
          border: "1px solid rgba(34,211,238,0.14)",
          boxShadow: "0 10px 28px rgba(0,0,0,0.28), inset 0 1px 0 rgba(255,255,255,0.03)",
        }}
      >
        <div
          className="flex items-center gap-2 px-4 py-2.5"
          style={{ borderBottom: "1px solid rgba(255,255,255,0.05)", background: "rgba(255,255,255,0.02)" }}
        >
          <span className="loader-dots flex items-center gap-1 text-cyan-300">
            <span />
            <span />
            <span />
          </span>
          <span className="font-display text-[10px] font-bold uppercase tracking-[0.16em]" style={{ color: "#7dd3fc" }}>
            Live Analysis
          </span>
        </div>

        <div className="px-5 py-4">
          <div className="grid gap-2.5 sm:grid-cols-3">
            {stages.map((stage, index) => {
              const isActive = index === activeStage;
              const isComplete = index < activeStage;
              return (
                <div
                  key={stage.label}
                  className="rounded-xl border px-3 py-3 transition-all"
                  style={{
                    background: isActive ? "rgba(34,211,238,0.08)" : "rgba(255,255,255,0.02)",
                    borderColor: isActive
                      ? "rgba(34,211,238,0.28)"
                      : isComplete
                        ? "rgba(74,222,128,0.24)"
                        : "rgba(255,255,255,0.06)",
                  }}
                >
                  <div className="flex items-center gap-2">
                    <span
                      className={`h-2 w-2 rounded-full ${isActive ? "loader-beacon" : ""}`}
                      style={{ background: isComplete ? "#4ade80" : isActive ? "#22d3ee" : "#334155" }}
                    />
                    <span className="text-xs font-semibold text-slate-100">{stage.label}</span>
                  </div>
                  <p className="mt-2 text-[11px] leading-5 text-slate-400">{stage.description}</p>
                </div>
              );
            })}
          </div>

          <div className="mt-4 overflow-hidden rounded-xl border border-cyan-500/10 bg-cyan-500/5">
            <div className="flex items-center gap-3 px-4 py-3">
              <Loader2 size={14} className="animate-spin text-cyan-300" />
              <div>
                <p className="text-sm font-medium text-slate-100">{statusText || "Reviewing evidence and building the answer"}</p>
                <p className="text-[11px] text-slate-400">Grounding each response against the uploaded sources.</p>
              </div>
            </div>
            <div className="h-1 w-full bg-white/5">
              <div className="loader-track h-full w-full" />
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function HomePage() {
  const fileInputRef = useRef<HTMLInputElement | null>(null);
  const bottomRef = useRef<HTMLDivElement | null>(null);

  const [messages, setMessages] = useState<Message[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [uploadJob, setUploadJob] = useState<UploadJob | null>(null);
  const [indexedFiles, setIndexedFiles] = useState<string[]>([]);
  const [pendingFiles, setPendingFiles] = useState<FileList | null>(null);
  const [showUploadWarning, setShowUploadWarning] = useState(false);
  const [statusText, setStatusText] = useState("");
  const [inputFocused, setInputFocused] = useState(false);
  const [docs, setDocs] = useState<DocMeta[]>([]);
  const [selectedDocIds, setSelectedDocIds] = useState<string[]>([]);

  const refreshDocs = useCallback(async () => {
    try {
      const res = await fetch(`${API}/documents`);
      const data: DocMeta[] = await res.json();
      setDocs(data);
      setSelectedDocIds((prev) => (prev.length ? prev : data.slice(-1).map((d) => d.document_id)));
    } catch {}
  }, []);

  useEffect(() => {
    refreshDocs();
  }, [refreshDocs]);

  useEffect(() => {
    setMessages([{
      id: "welcome",
      role: "assistant",
      text: "Hello! I'm Flamingo — your AI-powered biomedical research assistant. Upload your research papers and I'll help you find answers with precise citations.",
    }]);
  }, []);

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [messages, statusText]);

  const pollJob = useCallback(async (jobId: string, files: string[]) => {
    const stepLabels: Record<string, string> = {
      ingesting: "Extracting text from PDFs…",
      chunking: "Splitting into semantic chunks…",
      vectorizing: "Building vector embeddings…",
    };

    const interval = setInterval(async () => {
      try {
        const res = await fetch(`${API}/upload/status/${jobId}`);
        const data = await res.json();

        if (data.status === "done") {
          clearInterval(interval);
          setIndexedFiles((prev) => [...new Set([...prev, ...files])]);
          setUploadJob({ jobId, files, status: "done" });
          refreshDocs();
          setMessages((prev) => [...prev, {
            id: `done-${jobId}`,
            role: "assistant",
            text: `✓ Indexed ${files.length} paper${files.length > 1 ? "s" : ""}. You can now ask questions about: ${files.join(", ")}.`,
          }]);
        } else if (data.status === "failed") {
          clearInterval(interval);
          setUploadJob({ jobId, files, status: "failed" });
        } else {
          setUploadJob({ jobId, files, status: "processing", step: stepLabels[data.step] ?? "Processing…" });
        }
      } catch {
        clearInterval(interval);
      }
    }, 2000);
  }, [refreshDocs]);

  async function uploadPDFs(selected: FileList | null) {
    if (!selected || selected.length === 0) return;

    setShowUploadWarning(false);

    const formData = new FormData();
    Array.from(selected).forEach((file) => formData.append("files", file));
    setUploadJob({ jobId: "", files: Array.from(selected).map((file) => file.name), status: "pending" });

    try {
      const res = await fetch(`${API}/upload`, { method: "POST", body: formData });
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail ?? "Upload failed");
      setUploadJob({ jobId: data.job_id, files: data.files, status: "pending" });
      pollJob(data.job_id, data.files);
    } catch (err: unknown) {
      setUploadJob(null);
      setMessages((prev) => [...prev, {
        id: `err-${Date.now()}`,
        role: "assistant",
        text: err instanceof Error ? err.message : "Upload failed.",
      }]);
    }

    setPendingFiles(null);
  }

  async function sendMessage(forceMessage?: string) {
    const text = (forceMessage ?? input).trim();
    if (!text || loading) return;

    setInput("");
    setMessages((prev) => [...prev, { id: `u-${Date.now()}`, role: "user", text }]);
    setLoading(true);
    setStatusText("");

    try {
      const history = messages
        .filter((message) => message.id !== "welcome")
        .slice(-6)
        .map((message) => ({ role: message.role, content: message.text }));

      const res = await fetch(`${API}/chat`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ message: text, document_ids: selectedDocIds, history }),
      });

      if (!res.body) throw new Error("No response body");

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = "";

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (!line.startsWith("data: ")) continue;
          const raw = line.slice(6).trim();
          if (raw === "[DONE]") break;

          try {
            const event = JSON.parse(raw);
            if (event.type === "status") {
              setStatusText(event.message);
            } else if (event.type === "done") {
              setStatusText("");
              setMessages((prev) => [...prev, {
                id: `a-${Date.now()}`,
                role: "assistant",
                text: event.answer,
                citations: event.citations ?? [],
                confidence: event.confidence as Confidence,
                traceId: event.trace_id,
                latency: event.latency,
              }]);
            }
          } catch {}
        }
      }
    } catch {
      setMessages((prev) => [...prev, {
        id: `err-${Date.now()}`,
        role: "assistant",
        text: "Backend not reachable. Make sure the Python server is running on port 8000.",
      }]);
    }

    setLoading(false);
    setStatusText("");
  }

  async function submitFeedback(msg: Message, vote: "up" | "down") {
    if (!msg.traceId || msg.feedback) return;

    setMessages((prev) => prev.map((message) => (
      message.id === msg.id ? { ...message, feedback: vote } : message
    )));

    try {
      await fetch(`${API}/feedback`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ trace_id: msg.traceId, question: "", answer: msg.text, vote }),
      });
    } catch {}
  }

  return (
    <div
      className="flex h-screen flex-col overflow-hidden"
      style={{
        background: "#040d1a",
        backgroundImage: `
          linear-gradient(rgba(6,182,212,0.04) 1px, transparent 1px),
          linear-gradient(90deg, rgba(6,182,212,0.04) 1px, transparent 1px)
        `,
        backgroundSize: "40px 40px",
      }}
    >
      {showUploadWarning && (
        <Modal title="Upload Research Papers" subtitle="Documents indexed in the background" accentFrom="from-cyan-600" accentTo="to-blue-700">
          <div className="flex items-start gap-3 rounded-xl border border-cyan-500/20 bg-cyan-500/5 p-3 text-sm text-cyan-300">
            <FileText size={14} className="mt-0.5 shrink-0" />
            <span><strong className="text-cyan-200">{pendingFiles?.length ?? 0} PDF{(pendingFiles?.length ?? 0) > 1 ? "s" : ""}</strong> selected. Indexing runs in the background — you can keep chatting while papers process.</span>
          </div>
          <div className="mt-5 flex justify-end gap-3">
            <button
              onClick={() => {
                setPendingFiles(null);
                setShowUploadWarning(false);
              }}
              className="rounded-xl border border-white/10 px-4 py-2 text-sm text-slate-400 transition-colors hover:bg-white/5"
            >
              Cancel
            </button>
            <button
              onClick={() => uploadPDFs(pendingFiles)}
              className="rounded-xl bg-gradient-to-r from-cyan-600 to-blue-700 px-5 py-2 text-sm font-medium text-white shadow-lg shadow-cyan-500/20 transition-opacity hover:opacity-90"
            >
              Upload & Index
            </button>
          </div>
        </Modal>
      )}

      <header
        className="relative shrink-0 border-b"
        style={{ background: "linear-gradient(135deg,#060f20 0%,#070d1c 100%)", borderColor: "rgba(6,182,212,0.12)" }}
      >
        <div className="absolute inset-x-0 bottom-0 h-[1px]" style={{ background: "linear-gradient(90deg, transparent, #06b6d4, #8b5cf6, #3b82f6, transparent)" }} />

        <div className="mx-auto flex max-w-5xl items-center justify-between px-6 py-4">
          <div className="flex items-center gap-3">
            <div
              className="flex h-10 w-10 items-center justify-center rounded-xl"
              style={{ background: "linear-gradient(135deg,#0e7490,#4f46e5)", boxShadow: "0 0 20px rgba(6,182,212,0.3)" }}
            >
              <Image src="/flamingo.png" alt="Flamingo" width={24} height={24} className="brightness-0 invert" />
            </div>
            <div>
              <h1 className="font-display text-lg font-bold tracking-tight text-white">Flamingo</h1>
              <p className="text-[11px] font-medium" style={{ color: "#22d3ee" }}>Biomedical Research Assistant</p>
            </div>
          </div>

          <div className="flex items-center gap-2">
            {docs.length > 0 && (
              <details className="relative">
                <summary
                  className="cursor-pointer list-none rounded-full px-3 py-1.5 text-xs font-medium"
                  style={{ background: "rgba(6,182,212,0.1)", border: "1px solid rgba(6,182,212,0.2)", color: "#67e8f9" }}
                >
                  {selectedDocIds.length} selected
                </summary>
                <div className="absolute right-0 z-50 mt-2 w-72 rounded-xl border border-white/10 bg-[#0d1a2e] p-2 shadow-2xl">
                  {docs.map((doc) => (
                    <label key={doc.document_id} className="flex items-center gap-2 rounded-lg px-2 py-1.5 text-xs text-slate-300 hover:bg-white/5">
                      <input
                        type="checkbox"
                        checked={selectedDocIds.includes(doc.document_id)}
                        onChange={(event) => setSelectedDocIds((prev) => (
                          event.target.checked
                            ? [...prev, doc.document_id]
                            : prev.filter((id) => id !== doc.document_id)
                        ))}
                      />
                      <span className="truncate">{doc.title}</span>
                      {doc.scanned && <span className="ml-auto rounded bg-amber-500/15 px-1 text-[10px] text-amber-400">OCR</span>}
                    </label>
                  ))}
                </div>
              </details>
            )}
            {indexedFiles.length > 0 && (
              <div
                className="flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium"
                style={{ background: "rgba(6,182,212,0.1)", border: "1px solid rgba(6,182,212,0.2)", color: "#67e8f9" }}
              >
                <FileText size={11} />
                {indexedFiles.length} paper{indexedFiles.length > 1 ? "s" : ""}
              </div>
            )}
            <div
              className="flex items-center gap-1.5 rounded-full px-3 py-1.5 text-xs font-medium"
              style={{ background: "rgba(52,211,153,0.1)", border: "1px solid rgba(52,211,153,0.2)", color: "#6ee7b7" }}
            >
              <span className="pulse-glow h-2 w-2 rounded-full bg-emerald-400" />
              AI Active
            </div>
            <button
              onClick={() => {
                setMessages([]);
                setInput("");
              }}
              className="rounded-full px-3 py-1.5 text-xs transition-all"
              style={{ border: "1px solid rgba(255,255,255,0.08)", color: "#64748b" }}
              onMouseEnter={(event) => {
                event.currentTarget.style.color = "#94a3b8";
                event.currentTarget.style.background = "rgba(255,255,255,0.05)";
              }}
              onMouseLeave={(event) => {
                event.currentTarget.style.color = "#64748b";
                event.currentTarget.style.background = "transparent";
              }}
            >
              Clear
            </button>
          </div>
        </div>
      </header>

      {uploadJob && uploadJob.status !== "done" && (
        <div className="mx-auto w-full max-w-5xl shrink-0 px-6 pt-3">
          <div
            className="flex items-center gap-3 rounded-xl px-4 py-3 text-sm"
            style={{
              background: uploadJob.status === "failed" ? "rgba(239,68,68,0.08)" : "rgba(6,182,212,0.07)",
              border: `1px solid ${uploadJob.status === "failed" ? "rgba(239,68,68,0.2)" : "rgba(6,182,212,0.2)"}`,
              color: uploadJob.status === "failed" ? "#f87171" : "#67e8f9",
            }}
          >
            {uploadJob.status === "failed"
              ? <AlertCircle size={14} className="shrink-0" />
              : <Loader2 size={14} className="shrink-0 animate-spin" />}
            <span className="text-xs font-medium">
              {uploadJob.status === "failed" ? "Indexing failed. Please try again." : uploadJob.step ?? `Queuing ${uploadJob.files.length} paper(s)…`}
            </span>
            {uploadJob.status === "processing" && (
              <div className="ml-auto h-1 w-24 overflow-hidden rounded-full" style={{ background: "rgba(6,182,212,0.15)" }}>
                <div className="shimmer h-full w-full rounded-full" />
              </div>
            )}
          </div>
        </div>
      )}

      {indexedFiles.length > 0 && (
        <div className="mx-auto w-full max-w-5xl shrink-0 px-6 pt-3">
          <div
            className="flex flex-wrap items-center gap-2 rounded-xl px-4 py-2.5"
            style={{ background: "rgba(52,211,153,0.05)", border: "1px solid rgba(52,211,153,0.15)" }}
          >
            <CheckCircle size={12} className="shrink-0 text-emerald-400" />
            <span className="text-xs font-medium text-emerald-400">Indexed:</span>
            {indexedFiles.map((file, index) => (
              <span
                key={index}
                className="flex items-center gap-1.5 rounded-full px-2.5 py-0.5 text-xs"
                style={{ background: "rgba(52,211,153,0.08)", border: "1px solid rgba(52,211,153,0.15)", color: "#6ee7b7" }}
              >
                <FileText size={9} />
                {file}
              </span>
            ))}
          </div>
        </div>
      )}

      <main className="flex-1 overflow-y-auto">
        <div className="mx-auto max-w-5xl px-6 py-6">
          {messages.length === 0 && (
            <div className="flex flex-col items-center justify-center py-20 text-center">
              <div className="relative mb-8">
                <div
                  className="flex h-20 w-20 items-center justify-center rounded-3xl"
                  style={{ background: "linear-gradient(135deg,#0e7490,#4f46e5)", boxShadow: "0 0 40px rgba(6,182,212,0.25), 0 0 80px rgba(79,70,229,0.15)" }}
                >
                  <Microscope size={34} className="text-white" />
                </div>
                <div
                  className="absolute -right-1 -top-1 flex h-7 w-7 items-center justify-center rounded-full"
                  style={{ background: "linear-gradient(135deg,#7c3aed,#2563eb)", boxShadow: "0 0 12px rgba(124,58,237,0.5)" }}
                >
                  <Sparkles size={13} className="text-white" />
                </div>
              </div>
              <h2 className="font-display text-2xl font-bold text-slate-100">Ready to analyse your research</h2>
              <p className="mt-2 max-w-sm text-sm leading-relaxed text-slate-500">
                Upload biomedical PDFs to get started. Ask questions, get cited answers, and explore your corpus.
              </p>
              <div className="mt-8 flex flex-wrap justify-center gap-2">
                {["What are the key findings?", "Compare treatment outcomes", "List study limitations", "Summarize methodology"].map((question) => (
                  <button
                    key={question}
                    onClick={() => sendMessage(question)}
                    className="rounded-full px-4 py-2 text-xs transition-all"
                    style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.08)", color: "#64748b" }}
                    onMouseEnter={(event) => {
                      const el = event.currentTarget;
                      el.style.borderColor = "rgba(6,182,212,0.3)";
                      el.style.color = "#22d3ee";
                      el.style.background = "rgba(6,182,212,0.05)";
                    }}
                    onMouseLeave={(event) => {
                      const el = event.currentTarget;
                      el.style.borderColor = "rgba(255,255,255,0.08)";
                      el.style.color = "#64748b";
                      el.style.background = "rgba(255,255,255,0.04)";
                    }}
                  >
                    {question}
                  </button>
                ))}
              </div>
            </div>
          )}

          <div className="space-y-5">
            {messages.map((msg) => (
              <div key={msg.id}>
                {msg.role === "user" && (
                  <div className="flex items-end justify-end gap-3">
                    <div
                      className="max-w-[68%] rounded-2xl rounded-br-sm px-5 py-3.5"
                      style={{ background: "linear-gradient(135deg,#1d4ed8,#4f46e5)", boxShadow: "0 4px 20px rgba(79,70,229,0.25)" }}
                    >
                      <p className="text-sm leading-relaxed text-white">{msg.text}</p>
                    </div>
                    <div
                      className="flex h-8 w-8 shrink-0 items-center justify-center rounded-full text-slate-400"
                      style={{ background: "rgba(255,255,255,0.06)", border: "1px solid rgba(255,255,255,0.08)" }}
                    >
                      <User size={15} />
                    </div>
                  </div>
                )}

                {msg.role === "assistant" && (
                  <div className="flex items-start gap-3">
                    <div
                      className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl"
                      style={{ background: "linear-gradient(135deg,#0e7490,#4f46e5)", boxShadow: "0 0 16px rgba(6,182,212,0.25)" }}
                    >
                      <Image src="/flamingo.png" alt="Flamingo" width={20} height={20} className="brightness-0 invert" />
                    </div>

                    <div
                      className="max-w-[82%] overflow-hidden rounded-2xl rounded-tl-sm"
                      style={{
                        background: "#0a1628",
                        border: "1px solid rgba(6,182,212,0.12)",
                        boxShadow: "0 4px 24px rgba(0,0,0,0.3), 0 0 0 1px rgba(6,182,212,0.05)",
                      }}
                    >
                      <div
                        className="flex items-center gap-2 px-4 py-2.5"
                        style={{ borderBottom: "1px solid rgba(255,255,255,0.05)", background: "rgba(255,255,255,0.02)" }}
                      >
                        <Sparkles size={11} style={{ color: "#a78bfa" }} />
                        <span className="font-display text-[10px] font-bold uppercase tracking-[0.2em]" style={{ color: "#a78bfa" }}>Flamingo AI</span>
                        {msg.latency !== undefined && (
                          <span className="ml-auto font-mono text-[10px] text-slate-600">{msg.latency.toFixed(2)}s</span>
                        )}
                      </div>

                      <div className="px-5 py-4">
                        <p className="whitespace-pre-wrap text-sm leading-relaxed text-slate-300">{msg.text}</p>
                        {msg.citations && msg.citations.length > 0 && <CitationsPanel citations={msg.citations} />}
                      </div>

                      {msg.confidence && (
                        <div
                          className="flex items-center gap-3 px-4 py-2.5"
                          style={{ borderTop: "1px solid rgba(255,255,255,0.05)", background: "rgba(255,255,255,0.01)" }}
                        >
                          <ConfidenceBadge level={msg.confidence} />
                          {msg.traceId && (
                            <span className="font-mono text-[10px]" style={{ color: "#1e3a5f" }}>#{msg.traceId}</span>
                          )}
                          {msg.traceId && (
                            <div className="ml-auto flex items-center gap-1">
                              <span className="mr-1 text-[10px] text-slate-600">Helpful?</span>
                              <button
                                onClick={() => submitFeedback(msg, "up")}
                                disabled={!!msg.feedback}
                                className="rounded-lg p-1.5 transition-all disabled:opacity-30"
                                style={msg.feedback === "up"
                                  ? { background: "rgba(52,211,153,0.15)", color: "#34d399" }
                                  : { color: "#475569" }}
                                onMouseEnter={(event) => {
                                  if (!msg.feedback) event.currentTarget.style.color = "#34d399";
                                }}
                                onMouseLeave={(event) => {
                                  if (!msg.feedback) event.currentTarget.style.color = "#475569";
                                }}
                              >
                                <ThumbsUp size={12} />
                              </button>
                              <button
                                onClick={() => submitFeedback(msg, "down")}
                                disabled={!!msg.feedback}
                                className="rounded-lg p-1.5 transition-all disabled:opacity-30"
                                style={msg.feedback === "down"
                                  ? { background: "rgba(248,113,113,0.15)", color: "#f87171" }
                                  : { color: "#475569" }}
                                onMouseEnter={(event) => {
                                  if (!msg.feedback) event.currentTarget.style.color = "#f87171";
                                }}
                                onMouseLeave={(event) => {
                                  if (!msg.feedback) event.currentTarget.style.color = "#475569";
                                }}
                              >
                                <ThumbsDown size={12} />
                              </button>
                            </div>
                          )}
                        </div>
                      )}
                    </div>
                  </div>
                )}
              </div>
            ))}

            {loading && <AssistantLoader statusText={statusText} />}

            <div ref={bottomRef} />
          </div>
        </div>
      </main>

      <div
        className="shrink-0"
        style={{ borderTop: "1px solid rgba(255,255,255,0.05)", background: "rgba(4,13,26,0.9)", backdropFilter: "blur(20px)" }}
      >
        <div className="mx-auto max-w-5xl px-6 py-4">
          <div
            className="flex items-center gap-3 rounded-2xl px-4 py-3 transition-all duration-200"
            style={{
              background: "#0a1628",
              border: `1px solid ${inputFocused ? "rgba(6,182,212,0.35)" : "rgba(255,255,255,0.07)"}`,
              boxShadow: inputFocused ? "0 0 0 3px rgba(6,182,212,0.08), 0 0 20px rgba(6,182,212,0.06)" : "none",
            }}
          >
            <button
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadJob?.status === "processing"}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl transition-all disabled:opacity-40"
              style={{ background: "rgba(255,255,255,0.04)", border: "1px solid rgba(255,255,255,0.07)", color: "#475569" }}
              onMouseEnter={(event) => {
                const el = event.currentTarget;
                el.style.color = "#22d3ee";
                el.style.borderColor = "rgba(6,182,212,0.3)";
                el.style.background = "rgba(6,182,212,0.07)";
              }}
              onMouseLeave={(event) => {
                const el = event.currentTarget;
                el.style.color = "#475569";
                el.style.borderColor = "rgba(255,255,255,0.07)";
                el.style.background = "rgba(255,255,255,0.04)";
              }}
            >
              {uploadJob?.status === "processing" ? <Loader2 size={15} className="animate-spin" /> : <Upload size={15} />}
            </button>
            <input
              type="file"
              accept="application/pdf"
              multiple
              hidden
              ref={fileInputRef}
              onChange={(event) => {
                setPendingFiles(event.target.files);
                setShowUploadWarning(true);
              }}
            />

            <input
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) sendMessage();
              }}
              onFocus={() => setInputFocused(true)}
              onBlur={() => setInputFocused(false)}
              placeholder="Ask anything about your biomedical papers…"
              className="flex-1 bg-transparent text-sm outline-none"
              style={{ color: "#e2e8f0", caretColor: "#22d3ee" }}
            />

            {input && (
              <button
                onClick={() => setInput("")}
                className="flex h-8 w-8 shrink-0 items-center justify-center rounded-lg transition-all"
                style={{ color: "#475569" }}
                onMouseEnter={(event) => {
                  event.currentTarget.style.color = "#94a3b8";
                }}
                onMouseLeave={(event) => {
                  event.currentTarget.style.color = "#475569";
                }}
              >
                <X size={14} />
              </button>
            )}

            <button
              onClick={() => sendMessage()}
              disabled={loading || !input.trim()}
              className="flex h-9 w-9 shrink-0 items-center justify-center rounded-xl transition-all disabled:opacity-30"
              style={{ background: "linear-gradient(135deg,#0891b2,#4f46e5)", boxShadow: "0 0 16px rgba(6,182,212,0.2)" }}
              onMouseEnter={(event) => {
                event.currentTarget.style.opacity = "0.85";
              }}
              onMouseLeave={(event) => {
                event.currentTarget.style.opacity = "1";
              }}
            >
              <Send size={15} className="text-white" />
            </button>
          </div>

          <p className="mt-2 text-center text-[11px]" style={{ color: "#1e3a5f" }}>
            Flamingo answers only from your uploaded papers · Not for clinical advice
          </p>
        </div>
      </div>
    </div>
  );
}
