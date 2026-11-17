import {Suspense, use, useEffect} from "react";
import {Route, Routes, useLocation} from "react-router";
import {Layout} from "./components/Layout";
import {Loading} from "./components/Loading";
import {wasmReady} from "./lib/wasm";
import Home from "./pages/Home";
import LessonPage from "./pages/LessonPage";
import NotFound from "./pages/NotFound";

function ScrollToTop() {
    const {pathname, hash} = useLocation();
    useEffect(() => {
        if (hash) {
            document.getElementById(hash.slice(1))?.scrollIntoView();
        } else {
            window.scrollTo(0, 0);
        }
    }, [pathname, hash]);
    return null;
}

/** Suspends until the WebAssembly core is ready. */
function WasmGate({children}: {children: React.ReactNode}) {
    use(wasmReady);
    return children;
}

export default function App() {
    return (
        <Layout>
            <ScrollToTop />
            <Suspense fallback={<Loading label="Warming up the WebAssembly core" />}>
                <WasmGate>
                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/learn/:slug" element={<LessonPage />} />
                        <Route path="*" element={<NotFound />} />
                    </Routes>
                </WasmGate>
            </Suspense>
        </Layout>
    );
}
