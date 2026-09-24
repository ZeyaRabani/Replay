import { Clapperboard, LogOut, UserCircle } from "lucide-react";
import type { ReactNode } from "react";
import { Link, useNavigate } from "react-router-dom";
import { getUser, setUser } from "../api";

interface Props {
  children?: ReactNode;
}

export default function TopBar({ children }: Props) {
  const nav = useNavigate();
  const user = getUser();
  return (
    <header className="flex items-center gap-3 px-4 py-2 bg-zinc-900 border-b border-zinc-800">
      <Link to="/projects" className="flex items-center gap-2 font-semibold text-sm hover:text-amber-300">
        <Clapperboard size={18} className="text-amber-400" /> Replay Highlights
      </Link>
      <div className="flex items-center gap-2 min-w-0 flex-1">{children}</div>
      {user && (
        <div className="flex items-center gap-2 text-xs text-zinc-300">
          <UserCircle size={16} className="text-zinc-400" />
          <span className="font-medium">{user}</span>
          <button
            className="flex items-center gap-1 bg-zinc-800 hover:bg-zinc-700 rounded px-2 py-1 text-zinc-300"
            title="Switch user"
            onClick={() => {
              setUser(null);
              nav("/login");
            }}
          >
            <LogOut size={13} /> Switch user
          </button>
        </div>
      )}
    </header>
  );
}
