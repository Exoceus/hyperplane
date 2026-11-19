import {Component, type ErrorInfo, type ReactNode} from "react";
import {Link} from "react-router";
import {track} from "../lib/analytics";
import {isChunkLoadError, reloadForNewDeploy} from "../lib/staleDeploy";
import {Loading} from "./Loading";

interface State {
    error: Error | null;
    reloading: boolean;
}

/**
 * Catches anything a page throws, so a failure never blanks the whole site.
 * A missing code chunk (the site was redeployed while this tab was open)
 * reloads onto the new version; anything else shows a way out. Keyed by
 * pathname in App, so navigating away resets it.
 */
export class PageBoundary extends Component<{children: ReactNode}, State> {
    state: State = {error: null, reloading: false};

    static getDerivedStateFromError(error: Error): Partial<State> {
        return {error};
    }

    componentDidCatch(error: Error, info: ErrorInfo) {
        const stale = isChunkLoadError(error);
        track("error", {kind: stale ? "stale-chunk" : "render", message: error.message.slice(0, 200), path: window.location.pathname});
        if (!stale) console.error(error, info.componentStack);
        if (stale && reloadForNewDeploy()) this.setState({reloading: true});
    }

    render() {
        const {error, reloading} = this.state;
        if (!error) return this.props.children;
        if (reloading) return <Loading label="Loading the latest version" />;
        const stale = isChunkLoadError(error);
        return (
            <div className="not-found">
                <div className="eyebrow">{stale ? "Just updated" : "Something broke"}</div>
                <h1>{stale ? "A newer version of the site is out." : "This page hit an error."}</h1>
                <p>
                    {stale
                        ? "Hyperplane was updated while this tab was open. Reload to get the latest version."
                        : "Reloading usually fixes it. If it keeps happening, please open an issue on GitHub."}
                </p>
                {!stale && (
                    <p className="mono note-muted" style={{fontSize: 12.5}}>
                        {error.message}
                    </p>
                )}
                <div className="hero-ctas" style={{justifyContent: "center"}}>
                    <button type="button" className="btn btn-primary" onClick={() => window.location.reload()}>
                        Reload
                    </button>
                    <Link className="btn btn-ghost" to="/" onClick={() => this.setState({error: null})}>
                        All playgrounds
                    </Link>
                </div>
            </div>
        );
    }
}
