import {Link} from "react-router";

export default function NotFound() {
    return (
        <div className="not-found">
            <div className="eyebrow">404</div>
            <h1>This point is off the plane.</h1>
            <p>The page you are looking for doesn’t exist (yet).</p>
            <Link className="btn btn-primary" to="/">
                Back to lessons
            </Link>
        </div>
    );
}
