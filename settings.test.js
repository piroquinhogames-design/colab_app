import { describe, expect, it } from "vitest";
import { restoreLastSettings } from "./static/settings.js";

function restore(loras) {
  const state = { limits: { maxLoras: 8 }, selectedLoras: [] };
  restoreLastSettings({ loras }, {
    query: () => null,
    state,
    renderSelectedLoras: () => {},
    setMode: () => {},
    setEditLevel: () => {},
    log: () => {},
  });
  return state.selectedLoras;
}

describe("saved LoRA selections", () => {
  it("restores locally trained safetensors", () => {
    const result = restore([{ local_file: "anima-style.safetensors", name: "Anima Style", weight: 0.75 }]);
    expect(result).toHaveLength(1);
    expect(result[0].local_file).toBe("anima-style.safetensors");
    expect(result[0].version_id).toBeNull();
    expect(result[0].weight).toBe(0.75);
  });

  it("continues restoring remote Civitai LoRAs", () => {
    const result = restore([{ version_id: 12345, model_id: 678, name: "Remote", weight: 0.8 }]);
    expect(result[0].version_id).toBe(12345);
    expect(result[0].model_id).toBe(678);
  });

  it("rejects unsafe local filenames", () => {
    expect(restore([{ local_file: "../outside.safetensors", name: "Unsafe" }])).toHaveLength(0);
  });
});
