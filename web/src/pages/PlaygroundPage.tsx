import {Suspense, useEffect} from "react";
import {Link, useParams} from "react-router";
import {Loading} from "../components/Loading";
import {CATEGORIES, ORDERED, PLAYGROUND_COMPONENTS, PLAYGROUNDS} from "../playgrounds/registry";
import NotFound from "./NotFound";

export default function PlaygroundPage() {
    const {slug = ""} = useParams();
    const meta = PLAYGROUNDS.find((p) => p.slug === slug && p.status === "live");

    useEffect(() => {
        if (meta) document.title = `${meta.title} · Hyperplane`;
    }, [meta]);

    if (!meta) return <NotFound />;
    const Body = PLAYGROUND_COMPONENTS[slug];
    const idx = ORDERED.indexOf(meta);
    const prev = ORDERED[idx - 1];
    const next = ORDERED[idx + 1];
    const category = CATEGORIES.find((c) => c.id === meta.category);

    return (
        <article className="pg">
            <header className="pg-head">
                <div className="pg-crumbs">
                    <Link to="/">Playgrounds</Link>
                    <span aria-hidden>/</span>
                    <span>{category?.title}</span>
                </div>
                <div className="pg-title-row">
                    <div>
                        <h1>{meta.title}</h1>
                        <p className="pg-tagline">{meta.tagline}</p>
                    </div>
                    <nav className="pg-step" aria-label="Previous and next playground">
                        {prev ? (
                            <Link to={`/play/${prev.slug}`} className="pg-step-link" title={`Previous: ${prev.title}`}>
                                ← <span>{prev.title}</span>
                            </Link>
                        ) : null}
                        {next ? (
                            <Link to={`/play/${next.slug}`} className="pg-step-link" title={`Next: ${next.title}`}>
                                <span>{next.title}</span> →
                            </Link>
                        ) : null}
                    </nav>
                </div>
            </header>
            <Suspense fallback={<Loading label="Loading playground" />}>
                <Body key={slug} />
            </Suspense>
            <nav className="pg-nav" aria-label="More playgrounds">
                {prev ? (
                    <Link to={`/play/${prev.slug}`} className="pg-nav-link">
                        <span className="pg-nav-dir">← {prev.title}</span>
                    </Link>
                ) : (
                    <span />
                )}
                <Link to="/" className="pg-nav-link pg-nav-all">
                    All playgrounds
                </Link>
                {next ? (
                    <Link to={`/play/${next.slug}`} className="pg-nav-link next">
                        <span className="pg-nav-dir">{next.title} →</span>
                    </Link>
                ) : (
                    <span />
                )}
            </nav>
        </article>
    );
}
