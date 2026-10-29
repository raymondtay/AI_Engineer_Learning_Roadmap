import random

import torch


def test_sandbox(lab):
    assert lab.run_tool("37*48") == "1776"
    assert lab.run_tool("12+3-4") == "11"
    for bad in ["__import__('os').system('ls')", "2**100000", "open('x')", "37/0", "x*2"]:
        assert lab.run_tool(bad) == "ERR"


def test_env_tokens_masked(lab):
    class Scripted(torch.nn.Module):
        """Emits '<', then copies the tool result, then newline."""

        def __init__(self, script):
            super().__init__()
            self.script, self.i = script, 0

        def forward(self, x):
            logits = torch.full((1, x.shape[1], 256), -1e9)
            logits[0, -1, self.script[self.i]] = 0.0
            self.i += 1
            return logits, None

    m = Scripted([ord(c) for c in "<1776\n"])
    ids, mask, used, ans = lab.rollout(m, "37*48=", torch.Generator().manual_seed(0))
    text = bytes(ids).decode()
    assert text == "37*48=<>1776;1776\n" and used
    # prompt and env tokens are 0, policy tokens are 1
    assert mask == [0] * 6 + [1] + [0] * 6 + [1] * 5
    assert lab.verify("37*48=", ans) == 1.0


def test_rl_learns_to_use_tool(lab):
    s = lab.train(steps=20)
    k = 5
    assert sum(s["tool_rate"][-k:]) / k > sum(s["tool_rate"][:k]) / k + 0.15
    assert sum(s["reward"][-k:]) / k > sum(s["reward"][:k]) / k
