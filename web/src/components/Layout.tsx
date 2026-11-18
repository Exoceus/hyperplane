import type {ReactNode} from "react";
import {Link} from "react-router";
import {PlaygroundMenu} from "./PlaygroundMenu";

const REPO = "https://github.com/Exoceus/hyperplane";

export function Logo({size = 22}: {size?: number}) {
    return (
        <span className="logo" style={{fontSize: size}}>
            <svg className="logo-mark" viewBox="0 0 32 32" width={size * 1.2} height={size * 1.2} aria-hidden>
                <rect x="1" y="1" width="30" height="30" rx="7" fill="#1a1916" stroke="#35322c" />
                <circle cx="10" cy="20" r="2.6" fill="#f0ac5f" />
                <circle cx="13.5" cy="25" r="2.2" fill="#f0ac5f" opacity="0.8" />
                <circle cx="21" cy="9.5" r="2.6" fill="#58c4dd" />
                <circle cx="25" cy="14" r="2.2" fill="#58c4dd" opacity="0.8" />
                <path d="M6 27.5 L26.5 5" stroke="#f1eee6" strokeWidth="2.2" strokeLinecap="round" />
            </svg>
            <span className="logo-word">hyperplane</span>
        </span>
    );
}

export function Layout({children}: {children: ReactNode}) {
    return (
        <div className="app">
            <header className="site-header">
                <div className="site-header-inner">
                    <Link to="/" className="site-brand" aria-label="Hyperplane home">
                        <Logo />
                    </Link>
                    <nav className="site-nav">
                        <PlaygroundMenu />
                        <a href={REPO} target="_blank" rel="noreferrer">
                            GitHub
                        </a>
                    </nav>
                </div>
            </header>
            <main className="site-main">{children}</main>
            <footer className="site-footer">
                <div className="site-footer-inner">
                    <Logo size={18} />
                    <p>
                        Every figure is computed live in your browser by a Rust core compiled to WebAssembly. No servers, no tracking. Just
                        math.
                    </p>
                    <a href={REPO} target="_blank" rel="noreferrer">
                        Source on GitHub ↗
                    </a>
                </div>
            </footer>
        </div>
    );
}
