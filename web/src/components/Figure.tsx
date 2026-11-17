import {Component, type ReactNode} from "react";

/** Keeps a failure inside one figure from taking down the whole lesson. */
class FigureBoundary extends Component<{children: ReactNode}, {error: Error | null}> {
    state = {error: null as Error | null};
    static getDerivedStateFromError(error: Error) {
        return {error};
    }
    render() {
        if (this.state.error)
            return (
                <div className="figure-error">
                    This figure hit an error: <code>{this.state.error.message}</code>.{" "}
                    <button type="button" className="btn btn-ghost" onClick={() => this.setState({error: null})}>
                        Try again
                    </button>
                </div>
            );
        return this.props.children;
    }
}

interface FigureProps {
    children: ReactNode;
    /** Short title shown in the figure header. */
    title?: ReactNode;
    /** Figure number, e.g. "1". */
    n?: string | number;
    caption?: ReactNode;
    /** Hint shown in the header (e.g. "drag the arrow"). */
    hint?: ReactNode;
    className?: string;
}

/** A framed, interactive figure that breaks out of the prose column. */
export function Figure({children, title, n, caption, hint, className}: FigureProps) {
    return (
        <figure className={`figure ${className ?? ""}`}>
            {(title || hint) && (
                <header className="figure-head">
                    {title && (
                        <span className="figure-title">
                            {n !== undefined && <span className="figure-num">Fig. {n}</span>}
                            {title}
                        </span>
                    )}
                    {hint && <span className="figure-hint">{hint}</span>}
                </header>
            )}
            <div className="figure-body">
                <FigureBoundary>{children}</FigureBoundary>
            </div>
            {caption && <figcaption className="figure-caption">{caption}</figcaption>}
        </figure>
    );
}

/** Two-column layout inside a figure: visuals on the left, controls on the right. */
export function FigureLayout({main, side, sideWidth}: {main: ReactNode; side: ReactNode; sideWidth?: number}) {
    return (
        <div className="figure-layout" style={sideWidth ? ({"--side": `${sideWidth}px`} as React.CSSProperties) : undefined}>
            <div className="figure-main">{main}</div>
            <div className="figure-side">{side}</div>
        </div>
    );
}

/** A labeled panel over a plot (e.g. "data space" / "weight space"). */
export function Panel({label, children, right}: {label?: ReactNode; children: ReactNode; right?: ReactNode}) {
    return (
        <div className="panel">
            {(label || right) && (
                <div className="panel-label">
                    <span>{label}</span>
                    {right && <span className="panel-label-right">{right}</span>}
                </div>
            )}
            {children}
        </div>
    );
}

export function Callout({kind = "insight", title, children}: {kind?: "insight" | "try" | "note"; title?: ReactNode; children: ReactNode}) {
    const defaultTitle = kind === "try" ? "Try it" : kind === "note" ? "Note" : "Key idea";
    return (
        <aside className={`callout callout-${kind}`}>
            <div className="callout-title">{title ?? defaultTitle}</div>
            <div className="callout-body">{children}</div>
        </aside>
    );
}
