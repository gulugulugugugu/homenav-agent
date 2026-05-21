import "./App.css";

function App() {
  const demos = [
    {
      title: "Navigate to Sofa",
      command: "请到沙发旁边",
      video: "/demos/sofa_demo.mp4",
    },
    {
      title: "Navigate to Bed",
      command: "请到床旁边",
      video: "/demos/bed_demo.mp4",
    },
    {
      title: "Navigate to Table",
      command: "请到桌子旁边",
      video: "/demos/table_demo.mp4",
    },
    {
      title: "Navigate to Chair",
      command: "请到椅子旁边",
      video: "/demos/chair_demo.mp4",
    },
  ];

  return (
    <main className="page">
      <section className="hero">
        <h1>HomeNav-Agent</h1>
        <p>
          A language-guided embodied navigation agent in Habitat simulation.
        </p>

        <div className="tags">
          <span>Embodied AI</span>
          <span>Habitat-Sim</span>
          <span>RGB-D Navigation</span>
          <span>Language-Guided Agent</span>
        </div>
      </section>

      <section className="section">
        <h2>Project Overview</h2>
        <p>
          HomeNav-Agent accepts text commands such as “请到沙发旁边”, parses the
          target location, runs an embodied navigation loop in Habitat
          simulation, and responds with “还需要什么？” after task completion.
        </p>
      </section>

      <section className="section">
        <h2>Demo Videos</h2>

        <div className="grid">
          {demos.map((demo) => (
            <div className="card" key={demo.title}>
              <h3>{demo.title}</h3>

              <p>
                <strong>User:</strong> {demo.command}
              </p>

              <p>
                <strong>Agent:</strong> 任务完成。还需要什么？
              </p>

              <video controls src={demo.video} />
            </div>
          ))}
        </div>
      </section>

      <section className="section">
        <h2>System Architecture</h2>

        <div className="pipeline">
          <span>User Command</span>
          <span>Language Parser</span>
          <span>Agent State Machine</span>
          <span>Visual Target Adapter</span>
          <span>Habitat Simulator</span>
          <span>Navigation Actions</span>
        </div>
      </section>

      <section className="section">
        <h2>No Privileged Information Policy</h2>
        <p>
          The final design only uses RGB-D observations and robot state. It does
          not rely on simulator object coordinates, ground-truth target poses,
          semantic scene graphs, or shortest-path oracles during execution.
        </p>
      </section>

      <section className="section">
        <h2>Current Implementation</h2>
        <p>
          The current demo validates the full language-to-navigation pipeline.
          The perception module is implemented as a replaceable visual-target
          adapter, which can later be upgraded to OWL-ViT, GroundingDINO, or
          YOLO-World.
        </p>
      </section>

      <section className="section">
        <h2>Future Work: Continual Learning</h2>
        <p>
          This part is a research proposal and is not implemented in the current
          demo. Future versions can collect human intervention data when the
          agent gets stuck or makes navigation mistakes, then use those
          correction trajectories to fine-tune the navigation policy while
          replaying older successful episodes to reduce catastrophic forgetting.
        </p>
      </section>
    </main>
  );
}

export default App;
