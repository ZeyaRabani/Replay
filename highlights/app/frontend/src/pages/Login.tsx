import { Clapperboard, Loader2, LogIn } from "lucide-react";
import { useEffect, useState } from "react";
import { useNavigate } from "react-router-dom";
import { setUser, usersApi } from "../api";
import type { User } from "../types";

export default function Login() {
  const nav = useNavigate();
  const [name, setName] = useState("");
  const [users, setUsers] = useState<User[]>([]);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    usersApi
      .list()
      .then(setUsers)
      .catch(() => undefined);
  }, []);

  const login = async (n: string) => {
    const trimmed = n.trim();
    if (!trimmed) return;
    setBusy(true);
    setError(null);
    try {
      const u = await usersApi.create(trimmed);
      setUser(u.name);
      nav("/projects", { replace: true });
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center p-6">
      <div className="w-full max-w-sm bg-zinc-900 border border-zinc-800 rounded-xl p-6 shadow-xl">
        <div className="flex items-center gap-2 font-semibold text-lg mb-1">
          <Clapperboard size={22} className="text-amber-400" /> Replay Highlights
        </div>
        <p className="text-sm text-zinc-400 mb-5">Enter a display name to see your projects. No password needed.</p>
        <form
          onSubmit={(e) => {
            e.preventDefault();
            void login(name);
          }}
          className="flex flex-col gap-3"
        >
          <input
            autoFocus
            className="bg-zinc-800 border border-zinc-700 rounded px-3 py-2 text-sm placeholder:text-zinc-500 focus:outline-none focus:border-amber-400"
            placeholder="your name"
            maxLength={40}
            value={name}
            onChange={(e) => setName(e.target.value)}
          />
          <button
            type="submit"
            disabled={busy || !name.trim()}
            className="flex items-center justify-center gap-2 bg-amber-500 hover:bg-amber-400 text-zinc-900 font-semibold rounded px-3 py-2 text-sm disabled:opacity-40"
          >
            {busy ? <Loader2 size={15} className="animate-spin" /> : <LogIn size={15} />} Continue
          </button>
          {error && <div className="text-xs text-red-300">{error}</div>}
        </form>
        {users.length > 0 && (
          <div className="mt-5">
            <div className="text-[11px] uppercase tracking-wide text-zinc-500 mb-2">Existing profiles</div>
            <div className="flex flex-wrap gap-2">
              {users.map((u) => (
                <button
                  key={u.name}
                  disabled={busy}
                  onClick={() => void login(u.name)}
                  className="bg-zinc-800 hover:bg-zinc-700 border border-zinc-700 rounded-full px-3 py-1 text-xs"
                >
                  {u.name}
                  {u.n_projects !== undefined && <span className="text-zinc-500"> · {u.n_projects}</span>}
                </button>
              ))}
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
