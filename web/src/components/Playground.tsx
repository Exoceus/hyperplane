import {Component, Fragment, useState, type ReactNode} from "react";
import {useSearchParams} from "react-router";

/** One preset situation inside a playground: a configured instrument plus what to try. */
export interface Scenario {
    id: string;
    title: string;
    /** One or two sentences: what to do, and what to look for. */
    prompt: ReactNode;
    render: () => ReactNode;
}

/** Keeps a failure inside one instrument from taking down the whole page. */
class StageBoundary extends Component<{children: ReactNode}, {error: Error | null}> {
    state = {error: null as Error | null};
    static getDerivedStateFromError(error: Error) {
        return {error};
    }
    render() {
        if (this.state.error)
            return (
                <div className="figure-error">
                    This playground hit an error: <code>{this.state.error.message}</code>.{" "}
                    <button type="button" className="btn btn-ghost" onClick={() => this.setState({error: null})}>
                        Try again
                    </button>
                </div>
            );
        return this.props.children;
    }
}

/** Frame for an interactive instrument, with an optional interaction hint. */
export function Instrument({hint, children}: {hint?: ReactNode; children: ReactNode}) {
    return (
        <div className="instrument">
            {hint && <div className="instrument-hint">{hint}</div>}
            {children}
        </div>
    );
}

interface PlaygroundProps {
    scenarios: Scenario[];
    /** Short "how it works" notes: a few bullets and the key formulas. */
    explainer: ReactNode;
}

/**
 * The playground shell: a row of scenario presets above one big interactive
 * stage. The selected scenario is kept in the URL (?s=…) so it can be shared.
 */
export function Playground({scenarios, explainer}: PlaygroundProps) {
    const [params, setParams] = useSearchParams();
    const [nonce, setNonce] = useState(0);
    const active = scenarios.find((s) => s.id === params.get("s")) ?? scenarios[0];

    const choose = (id: string) => {
        setParams(id === scenarios[0].id ? {} : {s: id}, {replace: true, preventScrollReset: true});
        setNonce((n) => n + 1);
    };

    return (
        <>
            <nav className="scenarios" aria-label="Scenarios">
                {scenarios.map((s, i) => (
                    <button key={s.id} type="button" className={`scenario ${s.id === active.id ? "is-active" : ""}`} onClick={() => choose(s.id)} aria-pressed={s.id === active.id}>
                        <span className="scenario-num">{String(i + 1).padStart(2, "0")}</span>
                        <span className="scenario-title">{s.title}</span>
                    </button>
                ))}
            </nav>
            <div className="scenario-prompt">
                <span className="scenario-prompt-label">Try this</span>
                <span className="scenario-prompt-text">{active.prompt}</span>
                <button type="button" className="btn btn-subtle scenario-reset" onClick={() => setNonce((n) => n + 1)} title="Reset this scenario">
                    Reset
                </button>
            </div>
            <section className="stage">
                <StageBoundary key={`${active.id}-${nonce}`}>
                    <Fragment key={`${active.id}-${nonce}`}>{active.render()}</Fragment>
                </StageBoundary>
            </section>
            <details className="explainer">
                <summary>How it works</summary>
                <div className="explainer-body">{explainer}</div>
            </details>
        </>
    );
}
