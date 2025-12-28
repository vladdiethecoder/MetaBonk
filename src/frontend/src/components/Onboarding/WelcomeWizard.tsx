import { useEffect, useState } from "react";
import Joyride, { STATUS, type CallBackProps, type Step } from "react-joyride";
import { isTauri } from "../../lib/tauri";

const steps: Step[] = [
  {
    target: ".topbar",
    content: "Welcome to MetaBonk. This is your global command bar for navigation and system status.",
    placement: "bottom",
  },
  {
    target: ".omega-ctl",
    content: "Use the Omega control to start or stop training/gameplay without touching the CLI.",
    placement: "bottom",
  },
  {
    target: ".nav",
    content: "Navigate between Lobby, Neural Interface, Laboratory, and Codex.",
    placement: "bottom",
  },
];

export default function WelcomeWizard() {
  const [run, setRun] = useState(false);

  useEffect(() => {
    if (!isTauri()) return;
    try {
      const onboarded = window.localStorage.getItem("mb:onboarded");
      if (!onboarded) return;
      const seen = window.localStorage.getItem("mb:seenIntro");
      if (!seen) setRun(true);
    } catch {
      setRun(true);
    }
  }, []);

  const handle = (data: CallBackProps) => {
    if ([STATUS.FINISHED, STATUS.SKIPPED].includes(data.status)) {
      try {
        window.localStorage.setItem("mb:seenIntro", "1");
      } catch {}
      setRun(false);
    }
  };

  return (
    <Joyride
      steps={steps}
      run={run}
      continuous
      showSkipButton
      callback={handle}
      styles={{
        options: {
          zIndex: 1000,
          backgroundColor: "#0b1015",
          textColor: "#e7f6ff",
          primaryColor: "#7bffe6",
        },
      }}
    />
  );
}
