import {
  createContext,
  useContext,
  useEffect,
  useState,
  type ReactNode,
} from "react";

export type LayoutMode = "auto" | "desktop" | "mobile";
const KEY = "hl.layout";
const MQ = "(max-width: 767px)";

interface Layout {
  mode: LayoutMode;
  setMode: (m: LayoutMode) => void;
  isMobile: boolean;
}

const Ctx = createContext<Layout>({ mode: "auto", setMode: () => {}, isMobile: false });

function readMode(): LayoutMode {
  const v = localStorage.getItem(KEY);
  return v === "desktop" || v === "mobile" ? v : "auto";
}

export function LayoutProvider({ children }: { children: ReactNode }) {
  const [mode, setModeState] = useState<LayoutMode>(readMode);
  const [narrow, setNarrow] = useState(() => window.matchMedia(MQ).matches);

  useEffect(() => {
    const mq = window.matchMedia(MQ);
    const on = () => setNarrow(mq.matches);
    mq.addEventListener("change", on);
    return () => mq.removeEventListener("change", on);
  }, []);

  const setMode = (m: LayoutMode) => {
    localStorage.setItem(KEY, m);
    setModeState(m);
  };

  const isMobile = mode === "mobile" || (mode === "auto" && narrow);

  useEffect(() => {
    const el = document.documentElement;
    el.dataset.layout = isMobile ? "mobile" : "desktop";
    el.classList.toggle("mobile", isMobile);
    // forced desktop on a phone gets a wide, zoomable viewport
    const meta = document.querySelector<HTMLMetaElement>('meta[name="viewport"]');
    if (meta) {
      meta.content =
        mode === "desktop" && narrow
          ? "width=1200"
          : "width=device-width, initial-scale=1";
    }
  }, [isMobile, mode, narrow]);

  return <Ctx.Provider value={{ mode, setMode, isMobile }}>{children}</Ctx.Provider>;
}

export const useLayout = () => useContext(Ctx);
