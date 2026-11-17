import {Suspense, useEffect} from "react";
import {Link, useParams} from "react-router";
import {Loading} from "../components/Loading";
import {LESSON_COMPONENTS, LESSONS, LIVE_LESSONS, TRACKS} from "../lessons/registry";
import NotFound from "./NotFound";

export default function LessonPage() {
    const {slug = ""} = useParams();
    const meta = LESSONS.find((l) => l.slug === slug && l.status === "live");

    useEffect(() => {
        if (meta) document.title = `${meta.title} — Hyperplane`;
    }, [meta]);

    if (!meta) return <NotFound />;
    const Body = LESSON_COMPONENTS[slug];
    const idx = LIVE_LESSONS.indexOf(meta);
    const prev = LIVE_LESSONS[idx - 1];
    const next = LIVE_LESSONS[idx + 1];
    const track = TRACKS.find((t) => t.id === meta.track);

    return (
        <article className="lesson">
            <header className="lesson-head">
                <div className="kicker">
                    <Link to="/#lessons">Lessons</Link> <span aria-hidden>/</span> {track?.title} <span aria-hidden>·</span> Lesson {idx + 1}
                </div>
                <h1>{meta.title}</h1>
                <p className="dek">{meta.tagline}</p>
                <div className="lesson-meta">
                    <span>{meta.minutes} min read</span>
                    <span aria-hidden>·</span>
                    <span>Interactive</span>
                </div>
            </header>
            <div className="lesson-body">
                <Suspense fallback={<Loading label="Loading lesson" />}>
                    <Body key={slug} />
                </Suspense>
            </div>
            <nav className="lesson-nav" aria-label="More lessons">
                {prev ? (
                    <Link to={`/learn/${prev.slug}`} className="lesson-nav-link prev">
                        <span className="lesson-nav-dir">← Previous</span>
                        <span className="lesson-nav-title">{prev.title}</span>
                    </Link>
                ) : (
                    <span />
                )}
                {next ? (
                    <Link to={`/learn/${next.slug}`} className="lesson-nav-link next">
                        <span className="lesson-nav-dir">Next →</span>
                        <span className="lesson-nav-title">{next.title}</span>
                    </Link>
                ) : (
                    <Link to="/#lessons" className="lesson-nav-link next">
                        <span className="lesson-nav-dir">You made it →</span>
                        <span className="lesson-nav-title">Back to all lessons</span>
                    </Link>
                )}
            </nav>
        </article>
    );
}
