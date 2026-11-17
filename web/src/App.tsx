import {Suspense, use, useEffect} from "react";
import {Navigate, Route, Routes, useLocation, useParams} from "react-router";
import {Layout} from "./components/Layout";
import {Loading} from "./components/Loading";
import {wasmReady} from "./lib/wasm";
import Home from "./pages/Home";
import NotFound from "./pages/NotFound";
import PlaygroundPage from "./pages/PlaygroundPage";

function ScrollToTop() {
    const {pathname} = useLocation();
    useEffect(() => {
        window.scrollTo(0, 0);
    }, [pathname]);
    return null;
}

/** Suspends until the WebAssembly core is ready. */
function WasmGate({children}: {children: React.ReactNode}) {
    use(wasmReady);
    return children;
}

/** Old /learn/:slug links keep working. */
function LegacyRedirect() {
    const {slug} = useParams();
    return <Navigate to={`/play/${slug}`} replace />;
}

export default function App() {
    return (
        <Layout>
            <ScrollToTop />
            <Suspense fallback={<Loading label="Warming up the WebAssembly core" />}>
                <WasmGate>
                    <Routes>
                        <Route path="/" element={<Home />} />
                        <Route path="/play/:slug" element={<PlaygroundPage />} />
                        <Route path="/learn/:slug" element={<LegacyRedirect />} />
                        <Route path="*" element={<NotFound />} />
                    </Routes>
                </WasmGate>
            </Suspense>
        </Layout>
    );
}
