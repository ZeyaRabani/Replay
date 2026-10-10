import { Pencil } from "lucide-react";
import { useEffect, useRef, useState } from "react";

/** Inline project-title editor: a small pencil that swaps to an input.
 * Enter or blur saves (via onSave), Esc cancels. */
export default function TitleEdit({
  value,
  onSave,
}: {
  value: string;
  onSave: (title: string) => void | Promise<void>;
}) {
  const [editing, setEditing] = useState(false);
  const [draft, setDraft] = useState(value);
  const ref = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (editing) {
      setDraft(value);
      ref.current?.focus();
      ref.current?.select();
    }
  }, [editing]); // eslint-disable-line react-hooks/exhaustive-deps

  if (!editing) {
    return (
      <button
        className="text-zinc-500 hover:text-amber-300 p-1 shrink-0"
        title="Rename project"
        onClick={() => setEditing(true)}
      >
        <Pencil size={12} />
      </button>
    );
  }
  const save = () => {
    const t = draft.trim();
    setEditing(false);
    if (t && t !== value) void onSave(t);
  };
  return (
    <input
      ref={ref}
      className="bg-zinc-800 border border-zinc-700 rounded px-1.5 py-0.5 text-sm text-zinc-100 w-full max-w-[16rem]"
      value={draft}
      maxLength={120}
      onChange={(e) => setDraft(e.target.value)}
      onBlur={save}
      onKeyDown={(e) => {
        if (e.key === "Enter") save();
        else if (e.key === "Escape") setEditing(false);
      }}
    />
  );
}
