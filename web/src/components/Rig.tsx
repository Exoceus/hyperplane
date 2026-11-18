import {useContext, type ReactNode} from "react";
import {Icon} from "./Controls";
import {ScenarioContext} from "./scenario";

interface RigProps {
    /** What to look at: one plot, or several side by side in a `.duo`. */
    visual: ReactNode;
    /** What is happening right now. Shown under the visual. */
    caption?: ReactNode;
    /** How to interact with the visual itself (drag, click, hover). */
    hint?: ReactNode;
    /**
     * Every control, in one place: the transport (Player / LiveControls)
     * first, then `<Group>`s of parameters.
     */
    controls?: ReactNode;
    /** Read-only outputs: stats, charts, legends, short notes. */
    readouts?: ReactNode;
}

/**
 * The layout every instrument shares. Things you look at sit on the left;
 * everything you can change is grouped on the right in one control deck,
 * with the readouts underneath it.
 */
export function Rig({visual, caption, hint, controls, readouts}: RigProps) {
    const scenario = useContext(ScenarioContext);
    return (
        <div className="rig">
            <div className="rig-visual">
                {visual}
                {(caption || hint) && (
                    <div className="rig-caption">
                        {caption && <div className="rig-status">{caption}</div>}
                        {hint && (
                            <div className="rig-hint">
                                <svg viewBox="0 0 24 24" width="14" height="14" aria-hidden>
                                    <path
                                        d="M9 11V5.5a1.5 1.5 0 0 1 3 0V10m0-1.5a1.5 1.5 0 0 1 3 0V11m0-1a1.5 1.5 0 0 1 3 0v4.5a6.5 6.5 0 0 1-6.5 6.5h-.6a6 6 0 0 1-4.7-2.3L4.6 15a1.5 1.5 0 0 1 2.3-1.9L9 15"
                                        fill="none"
                                        stroke="currentColor"
                                        strokeWidth="1.6"
                                        strokeLinecap="round"
                                        strokeLinejoin="round"
                                    />
                                </svg>
                                <span>{hint}</span>
                            </div>
                        )}
                    </div>
                )}
            </div>
            <div className="rig-side">
                <div className="deck" role="group" aria-label="Controls">
                    {controls}
                    {scenario && (
                        <button type="button" className="deck-reset" onClick={scenario.reset} title="Put every control back the way this scenario started">
                            {Icon.reset}
                            Reset scenario
                        </button>
                    )}
                </div>
                {readouts && <div className="readouts">{readouts}</div>}
            </div>
        </div>
    );
}

/** A labelled cluster of related controls inside the deck, e.g. "Data" or "Model". */
export function Group({label, children}: {label: ReactNode; children: ReactNode}) {
    return (
        <div className="group" role="group" aria-label={typeof label === "string" ? label : undefined}>
            <div className="group-label">{label}</div>
            <div className="group-body">{children}</div>
        </div>
    );
}

/** A row of buttons inside a group. */
export function Actions({children}: {children: ReactNode}) {
    return <div className="control-row">{children}</div>;
}
