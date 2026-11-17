export function Loading({label = "Loading"}: {label?: string}) {
    return (
        <div className="loading" role="status">
            <span className="loading-dot" />
            {label}…
        </div>
    );
}
