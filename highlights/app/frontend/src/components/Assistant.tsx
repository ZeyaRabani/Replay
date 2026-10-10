import { Loader2, MessageCircle, Send, Trash2, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { useLocation } from "react-router-dom";
import { assistantApi, getUser } from "../api";
import type { AssistantAction } from "../api";

interface Msg {
  role: "user" | "assistant";
  content: string;
  actions?: AssistantAction[];
}

const QUICK = [
  "Why is my match stuck?",
  "How do I refresh YouTube cookies?",
  "How long will processing take?",
  "What does 'needs sync offsets' mean?",
];

const COOKIE_STEPS = [
  "1. Chrome → extensions → “Get cookies.txt LOCALLY” → Details → enable “Allow in Incognito”.",
  "2. Open an incognito window, sign in to youtube.com, play any video a few seconds.",
  "3. Click the extension → Export (Netscape format) → cookies.txt downloads.",
  "4. Close the incognito window right away.",
  "5. Replay → Projects page → “YouTube access” → paste file contents → Save.",
];

function storageKey(): string {
  return `replay.assistant.${getUser() ?? "anon"}`;
}

function loadHistory(): Msg[] {
  try {
    const raw = localStorage.getItem(storageKey());
    const arr = raw ? JSON.parse(raw) : [];
    return Array.isArray(arr) ? arr.slice(-40) : [];
  } catch {
    return [];
  }
}

export default function Assistant() {
  const loc = useLocation();
  const [open, setOpen] = useState(false);
  const [configured, setConfigured] = useState<boolean | null>(null);
  const [msgs, setMsgs] = useState<Msg[]>(loadHistory);
  const [input, setInput] = useState("");
  const [thinking, setThinking] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    assistantApi.status()
      .then((s) => setConfigured(s.configured))
      .catch(() => setConfigured(false));
  }, []);

  useEffect(() => {
    try {
      localStorage.setItem(storageKey(), JSON.stringify(msgs.slice(-40)));
    } catch { /* full/blocked */ }
  }, [msgs]);

  useEffect(() => {
    scrollRef.current?.scrollTo({ top: scrollRef.current.scrollHeight });
  }, [msgs, thinking]);

  const projectId = (loc.pathname.match(/^\/projects\/([^/]+)/) || [])[1];
  const page = { path: loc.pathname, project_id: projectId };

  const send = async (text?: string) => {
    const content = (text ?? input).trim();
    if (!content || thinking) return;
    setError(null);
    const next = [...msgs, { role: "user" as const, content }];
    setMsgs(next);
    setInput("");
    setThinking(true);
    try {
      const r = await assistantApi.chat(
        next.map((m) => ({ role: m.role, content: m.content })),
        page,
      );
      setMsgs([...next, { role: "assistant", content: r.reply, actions: r.actions }]);
      if (r.actions.some((a) => a.ok)) {
        window.dispatchEvent(new Event("replay:refresh"));
      }
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setInput(content); // keep the user's text
      setMsgs(msgs);
    } finally {
      setThinking(false);
    }
  };

  const clear = () => {
    setMsgs([]);
    setError(null);
    try {
      localStorage.removeItem(storageKey());
    } catch { /* ignore */ }
  };

  return (
    <div className="fixed bottom-4 right-4 z-50 flex flex-col items-end gap-2">
      {open && (
        <div className="flex h-[560px] w-[380px] max-w-[calc(100vw-2rem)] max-h-[calc(100dvh-6rem)] flex-col rounded-xl border border-zinc-700 bg-zinc-900 shadow-2xl sm:w-[380px] w-[calc(100vw-2rem)]">
          <div className="flex items-center justify-between border-b border-zinc-700 px-3 py-2">
            <div>
              <div className="text-sm font-semibold text-zinc-100">Replay assistant</div>
              <div className="text-[11px] text-zinc-500">
                Knows Replay inside out · can fix things
              </div>
            </div>
            <div className="flex items-center gap-1">
              <button
                className="rounded p-1 text-zinc-400 hover:text-zinc-200"
                title="Clear conversation"
                onClick={clear}
              >
                <Trash2 size={14} />
              </button>
              <button
                className="rounded p-1 text-zinc-400 hover:text-zinc-200"
                onClick={() => setOpen(false)}
              >
                <X size={15} />
              </button>
            </div>
          </div>

          <div ref={scrollRef} className="flex-1 overflow-y-auto px-3 py-2 space-y-3">
            {configured === false && msgs.length === 0 && (
              <div className="text-xs text-zinc-300 space-y-2">
                <p>
                  The assistant isn&apos;t connected to an AI model yet. The admin
                  needs to add an API key on the server (deploy/.env →
                  OPENAI_API_KEY or ANTHROPIC_API_KEY and{" "}
                  <code>docker compose up -d</code>).
                </p>
                <p className="text-zinc-400">Meanwhile — YouTube cookie steps:</p>
                <ol className="list-none space-y-0.5 text-zinc-400">
                  {COOKIE_STEPS.map((s) => <li key={s}>{s}</li>)}
                </ol>
                <a className="text-amber-400 hover:underline" href="/projects">
                  YouTube access panel (Projects page)
                </a>
              </div>
            )}
            {msgs.length === 0 && configured !== false && (
              <div className="flex flex-wrap gap-1.5 pt-2">
                {QUICK.map((q) => (
                  <button
                    key={q}
                    className="rounded-full border border-zinc-700 px-2.5 py-1 text-[11px] text-zinc-300 hover:border-amber-500 hover:text-amber-300 disabled:opacity-40"
                    onClick={() => void send(q)}
                    disabled={!configured}
                  >
                    {q}
                  </button>
                ))}
              </div>
            )}
            {msgs.map((m, i) => (
              <div key={i} className={m.role === "user" ? "text-right" : "text-left"}>
                <div
                  className={`inline-block max-w-[85%] rounded-lg px-3 py-1.5 text-sm whitespace-pre-wrap text-left ${
                    m.role === "user"
                      ? "bg-amber-500 text-zinc-950"
                      : "bg-zinc-800 text-zinc-100"
                  }`}
                >
                  {m.content}
                </div>
                {m.actions && m.actions.length > 0 && (
                  <div className="mt-1 flex flex-wrap gap-1">
                    {m.actions.map((a, j) => (
                      <span
                        key={j}
                        className={`rounded-full border px-2 py-0.5 text-[10px] ${
                          a.ok
                            ? "border-emerald-700 text-emerald-400"
                            : "border-red-800 text-red-400"
                        }`}
                      >
                        {a.ok ? "✓" : "✗"} {a.summary}
                      </span>
                    ))}
                  </div>
                )}
              </div>
            ))}
            {thinking && (
              <div className="text-left">
                <div className="inline-block rounded-lg bg-zinc-800 px-3 py-1.5 text-sm text-zinc-400">
                  <Loader2 size={13} className="inline animate-spin" /> thinking…
                </div>
              </div>
            )}
            {error && <div className="text-xs text-red-400">{error}</div>}
          </div>

          <div className="flex items-end gap-2 border-t border-zinc-700 p-2">
            <textarea
              className="max-h-24 min-h-[34px] flex-1 resize-none rounded-md border border-zinc-700 bg-zinc-950 px-2 py-1.5 text-sm text-zinc-100 outline-none focus:border-amber-500 disabled:opacity-50"
              rows={1}
              placeholder={!configured ? "Assistant not configured…" : "Ask about your matches…"}
              value={input}
              disabled={!configured}
              onChange={(e) => setInput(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  void send();
                }
              }}
            />
            <button
              className="rounded-md bg-amber-500 p-2 text-zinc-950 hover:bg-amber-400 disabled:opacity-40"
              disabled={!input.trim() || thinking || !configured}
              onClick={() => void send()}
            >
              <Send size={14} />
            </button>
          </div>
        </div>
      )}
      <button
        className="flex items-center gap-1.5 rounded-full bg-amber-500 px-3.5 py-2.5 text-sm font-semibold text-zinc-950 shadow-lg hover:bg-amber-400"
        onClick={() => setOpen((o) => !o)}
      >
        <MessageCircle size={16} /> Help
      </button>
    </div>
  );
}
