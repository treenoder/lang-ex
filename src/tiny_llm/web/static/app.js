const select = document.querySelector("#model");
const metadata = document.querySelector("#metadata");
const status = document.querySelector("#status");
const response = document.querySelector("#response");
let models = [];

function showMetadata() {
  const model = models.find(item => item.name === select.value);
  metadata.textContent = model
    ? `${model.architecture.toUpperCase()} · ${model.size} preset · ${model.formatted_parameters} parameters`
    : "No trained models found.";
}

async function loadModels() {
  models = await fetch("/api/models").then(result => result.json());
  select.replaceChildren(...models.map(model => new Option(model.name, model.name)));
  showMetadata();
}

select.addEventListener("change", showMetadata);
document.querySelector("#send").addEventListener("click", async () => {
  if (!select.value) return;
  const button = document.querySelector("#send");
  button.disabled = true;
  status.textContent = "Generating on CPU…";
  try {
    const result = await fetch("/api/chat", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify({
        model: select.value,
        prompt: document.querySelector("#prompt").value,
        max_new_tokens: Number(document.querySelector("#tokens").value),
        temperature: Number(document.querySelector("#temperature").value),
        top_k: Number(document.querySelector("#top-k").value)
      })
    });
    const body = await result.json();
    if (!result.ok) throw new Error(body.detail || "Request failed");
    response.textContent = body.text;
    status.textContent = `Generated with ${body.model}`;
  } catch (error) {
    status.textContent = "Error";
    response.textContent = error.message;
  } finally {
    button.disabled = false;
  }
});

loadModels().catch(error => { status.textContent = error.message; });

