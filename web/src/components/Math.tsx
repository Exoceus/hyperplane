import katex from "katex";
import "katex/dist/katex.min.css";
import {useMemo} from "react";
import {TEX_MACROS} from "../lib/theme";

function render(tex: string, display: boolean): string {
    return katex.renderToString(tex, {
        displayMode: display,
        throwOnError: false,
        macros: {...TEX_MACROS},
        strict: "ignore",
    });
}

/** Inline math: <M>{String.raw`w \cdot x`}</M> */
export function M({children}: {children: string}) {
    const html = useMemo(() => render(children, false), [children]);
    return <span className="math-inline" dangerouslySetInnerHTML={{__html: html}} />;
}

/** Display math, centered on its own line. */
export function MathBlock({children, label}: {children: string; label?: string}) {
    const html = useMemo(() => render(children, true), [children]);
    return (
        <div className="math-block">
            <div className="math-block-body" dangerouslySetInnerHTML={{__html: html}} />
            {label && <span className="math-block-label">{label}</span>}
        </div>
    );
}
