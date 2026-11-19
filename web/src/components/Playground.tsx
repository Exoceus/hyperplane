import {Component, Fragment, useMemo, useState, type ReactNode} from "react";
import {useParams, useSearchParams} from "react-router";
import {track} from "../lib/analytics";
import {ScenarioContext} from "./scenario";

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

interface PlaygroundProps {
    scenarios: Scenario[];
    /** Short "how it works" notes: a few bullets and the key formulas. */
    explainer: ReactNode;
}

/**
 * The playground shell: numbered scenario presets above one big interactive
 * stage. The selected scenario is kept in the URL (?s=…) so it can be shared.
 */
export function Playground({scenarios, explainer}: PlaygroundProps) {
    const [params, setParams] = useSearchParams();
    const {slug = ""} = useParams();
    const [nonce, setNonce] = useState(0);
    const active = scenarios.find((s) => s.id === params.get("s")) ?? scenarios[0];
    const index = scenarios.indexOf(active);
    const scenarioCtx = useMemo(() => ({reset: () => setNonce((n) => n + 1)}), []);

    const choose = (id: string) => {
        track("scenario", {playground: slug, scenario: id});
        setParams(id === scenarios[0].id ? {} : {s: id}, {replace: true, preventScrollReset: true});
        setNonce((n) => n + 1);
    };

    return (
        <>
            <nav className="scenarios" aria-label="Scenarios">
                <span className="scenarios-label">Scenarios</span>
                {scenarios.map((s, i) => (
                    <button key={s.id} type="button" className={`scenario ${s.id === active.id ? "is-active" : ""}`} onClick={() => choose(s.id)} aria-pressed={s.id === active.id}>
                        <span className="scenario-num">{i + 1}</span>
                        <span className="scenario-title">{s.title}</span>
                    </button>
                ))}
            </nav>
            <section className="stage" aria-label={active.title}>
                <header className="stage-head">
                    <span className="stage-num">{String(index + 1).padStart(2, "0")}</span>
                    <div className="stage-prompt">
                        <span className="stage-prompt-label">Try this</span> {active.prompt}
                    </div>
                </header>
                <ScenarioContext.Provider value={scenarioCtx}>
                    <StageBoundary key={`${active.id}-${nonce}`}>
                        <Fragment key={`${active.id}-${nonce}`}>{active.render()}</Fragment>
                    </StageBoundary>
                </ScenarioContext.Provider>
            </section>
            <details className="explainer">
                <summary>How it works</summary>
                <div className="explainer-body">{explainer}</div>
            </details>
        </>
    );
}
