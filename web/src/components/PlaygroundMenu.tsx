import {useEffect, useRef, useState} from "react";
import {Link, useLocation} from "react-router";
import {CATEGORIES, LIVE} from "../playgrounds/registry";

/** "Playgrounds ▾" in the site header: every playground, grouped by category, one click away. */
export function PlaygroundMenu() {
    const [open, setOpen] = useState(false);
    const ref = useRef<HTMLDivElement>(null);
    const {pathname} = useLocation();
    const current = pathname.startsWith("/play/") ? pathname.slice(6) : null;

    // Close on navigation, outside clicks and Escape.
    const [seenPath, setSeenPath] = useState(pathname);
    if (seenPath !== pathname) {
        setSeenPath(pathname);
        setOpen(false);
    }
    useEffect(() => {
        if (!open) return;
        const onDown = (e: PointerEvent) => {
            if (!ref.current?.contains(e.target as Node)) setOpen(false);
        };
        const onKey = (e: KeyboardEvent) => e.key === "Escape" && setOpen(false);
        document.addEventListener("pointerdown", onDown);
        document.addEventListener("keydown", onKey);
        return () => {
            document.removeEventListener("pointerdown", onDown);
            document.removeEventListener("keydown", onKey);
        };
    }, [open]);

    return (
        <div className="pg-menu" ref={ref}>
            <button type="button" className={`pg-menu-button ${open ? "is-open" : ""}`} aria-expanded={open} aria-haspopup="true" onClick={() => setOpen(!open)}>
                Playgrounds
                <svg viewBox="0 0 12 12" width="10" height="10" aria-hidden>
                    <path d="M2 4.5l4 3.5 4-3.5" fill="none" stroke="currentColor" strokeWidth="1.6" strokeLinecap="round" strokeLinejoin="round" />
                </svg>
            </button>
            {open && (
                <div className="pg-menu-panel" role="menu">
                    {CATEGORIES.map((c) => {
                        const items = LIVE.filter((p) => p.category === c.id);
                        if (!items.length) return null;
                        return (
                            <div key={c.id} className="pg-menu-group">
                                <div className="pg-menu-cat">{c.title}</div>
                                {items.map((p) => (
                                    <Link key={p.slug} to={`/play/${p.slug}`} role="menuitem" className={`pg-menu-item ${p.slug === current ? "is-current" : ""}`}>
                                        {p.title}
                                    </Link>
                                ))}
                            </div>
                        );
                    })}
                </div>
            )}
        </div>
    );
}
