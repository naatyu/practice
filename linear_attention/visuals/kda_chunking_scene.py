"""Captioned Manim lesson: from the KDA recurrence to a chunk error system.

Render from this directory with Manim 0.19.0:
    manim -qm --media_dir .render -o kda_chunking kda_chunking_scene.py KDAChunking

This uses Text rather than LaTeX so rendering does not require a TeX install.
The triangular-system formulation is an educational derivation of the
recurrence; it is not a reproduction of FLA's optimized WY kernels.
"""

from pathlib import Path

from manim import (
    DOWN,
    LEFT,
    RIGHT,
    UP,
    Arrow,
    Create,
    CurvedArrow,
    Dot,
    FadeIn,
    FadeOut,
    Indicate,
    Line,
    Rectangle,
    RoundedRectangle,
    Scene,
    SurroundingRectangle,
    Text,
    Transform,
    VGroup,
    config,
)

# isort: split
from chunking_example import teaching_example, verify_example

BG = "#10131B"
FG = "#ECEFF5"
MUTED = "#ADB6CA"
BLUE = "#69C9F0"
GOLD = "#FFCA68"
GREEN = "#73DEB2"
PURPLE = "#B99CFF"
RED = "#E9959D"


def words(text, size=26, color=FG, mono=False):
    result = Text(
        text,
        font="DejaVu Sans Mono" if mono else "DejaVu Sans",
        font_size=size,
        color=color,
        line_spacing=0.8,
    )
    if result.width > 12.7:
        result.scale_to_fit_width(12.7)
    return result


def tile(text, color=BLUE, width=1.6):
    box = RoundedRectangle(
        width=width,
        height=0.8,
        corner_radius=0.12,
        stroke_color=color,
        stroke_width=2,
        fill_color=color,
        fill_opacity=0.09,
    )
    label = words(text, 25, color)
    if label.width > width - 0.2:
        label.scale_to_fit_width(width - 0.2)
    return VGroup(box, label.move_to(box))


def grid(values, color=BLUE, cell=0.78, diagonal=True):
    rows, cols = len(values), len(values[0])
    boxes, labels = VGroup(), VGroup()
    for r, row in enumerate(values):
        for c, value in enumerate(row):
            active = c < r or (diagonal and c == r)
            position = ((c - (cols - 1) / 2) * cell, ((rows - 1) / 2 - r) * cell, 0)
            box = Rectangle(
                width=cell,
                height=cell,
                stroke_color=MUTED,
                stroke_width=0.7,
                fill_color=color,
                fill_opacity=0.15 if active else 0.015,
            ).move_to(position)
            label = words(str(value), 23, FG if active else MUTED, mono=True)
            if label.width > cell - 0.14:
                label.scale_to_fit_width(cell - 0.14)
            label.move_to(position)
            boxes.add(box)
            labels.add(label)
    return VGroup(boxes, labels)


class KDAChunking(Scene):
    def construct(self):
        self.camera.background_color = BG
        self.caption_mobject = None
        self.cues = []
        self.chapters = []
        verify_example()
        self.example = teaching_example()
        self.intro()
        self.memory_loop()
        self.toy_inputs()
        self.follow_decay()
        self.key_influence()
        self.dependencies()
        self.solve_errors()
        self.outputs_and_state()
        self.chunk_pipeline()
        self.practical_scope()
        self.write_captions()

    def chapter(self, number, title):
        if self.cues:
            self.cues[-1][1] = self.time
        if self.mobjects:
            self.play(*[FadeOut(m) for m in list(self.mobjects)], run_time=0.6)
        self.clear()
        self.caption_mobject = None
        self.chapters.append((self.time, title))
        heading = words(title, 34).to_edge(UP, buff=0.65).to_edge(LEFT, buff=0.7)
        progress = Line(
            (-6.4, 3.7, 0),
            (-6.4 + 12.8 * number / 9, 3.7, 0),
            color=BLUE,
            stroke_width=3,
        )
        self.play(FadeIn(heading), Create(progress), run_time=0.7)

    def caption(self, text, hold=5):
        if self.cues:
            self.cues[-1][1] = self.time
        target = words(text, 26).move_to((0, -3.2, 0))
        if self.caption_mobject is None:
            self.play(FadeIn(target), run_time=0.4)
            self.caption_mobject = target
        else:
            self.play(Transform(self.caption_mobject, target), run_time=0.4)
        self.cues.append([self.time, self.time + hold, text])
        self.wait(hold)

    def intro(self):
        title = words("Chunking Kimi Delta Attention", 44).move_to((0, 1.0, 0))
        subtitle = words(
            "From three memory updates to one block of equations", 27, MUTED
        )
        subtitle.next_to(title, DOWN, buff=0.45)
        tokens = VGroup(*[tile(f"token {i}") for i in range(3)]).arrange(
            RIGHT, buff=0.35
        )
        tokens.move_to((0, -0.9, 0))
        self.play(FadeIn(title), FadeIn(subtitle), run_time=1.2)
        self.play(*[FadeIn(t, shift=UP * 0.2) for t in tokens], run_time=1.0)
        self.caption(
            "Start with the recurrence you know.\nThen reorganize its dependencies without changing the result.",
            6,
        )

    def memory_loop(self):
        self.chapter(1, "1. Why the token loop is sequential")
        positions = [-5.4, -1.8, 1.8, 5.4]
        states = VGroup(
            *[
                tile(name, PURPLE, 1.5).move_to((x, 0.6, 0))
                for name, x in zip(["S_in", "S0", "S1", "S2"], positions, strict=True)
            ]
        )
        tokens = VGroup(
            *[
                words(f"token {i}", 25, BLUE).move_to((positions[i + 1], 1.8, 0))
                for i in range(3)
            ]
        )
        arrows = VGroup(
            *[
                Arrow(
                    states[i].get_right(),
                    states[i + 1].get_left(),
                    buff=0.12,
                    color=MUTED,
                )
                for i in range(3)
            ]
        )
        steps = (
            VGroup(
                *[
                    words(s, 27, c)
                    for s, c in [
                        ("decay", PURPLE),
                        ("predict", BLUE),
                        ("correct", GOLD),
                        ("read", GREEN),
                    ]
                ]
            )
            .arrange(RIGHT, buff=0.85)
            .move_to((0, -1.3, 0))
        )
        self.play(FadeIn(states), FadeIn(tokens), *[Create(a) for a in arrows])
        self.caption(
            "Each token first decays memory, predicts its value,\nwrites the prediction error, then reads the updated memory.",
            7,
        )
        self.play(FadeIn(steps))
        for i in range(3):
            self.play(
                Indicate(states[i], color=GOLD),
                Indicate(tokens[i], color=BLUE),
                run_time=0.8,
            )
            dot = Dot(arrows[i].get_start(), color=GOLD, radius=0.09)
            self.add(dot)
            self.play(dot.animate.move_to(arrows[i].get_end()), run_time=0.9)
            self.play(FadeOut(dot), Indicate(states[i + 1], color=PURPLE), run_time=0.6)
        self.caption(
            "The next prediction needs the state produced by the previous token.\nDrawing chunk boundaries around this loop does not parallelize it.",
            6,
        )
        boundary = SurroundingRectangle(
            VGroup(*states[1:], tokens), color=BLUE, buff=0.3
        )
        self.play(Create(boundary))
        self.wait(2)

    def toy_inputs(self):
        self.chapter(2, "2. Keep one small example in view")
        columns = [-5.5, -2.8, 0.4, 3.2, 5.4]
        headers = ["token", "key = query", "retention", "beta", "value"]
        header = VGroup(
            *[
                words(s, 22, MUTED).move_to((x, 1.8, 0))
                for s, x in zip(headers, columns, strict=True)
            ]
        )
        self.play(FadeIn(header))
        for t in range(3):
            row = [
                str(t),
                str(self.example.keys[t]),
                str(self.example.retention[t]),
                str(self.example.beta[t]),
                str(self.example.values[t]),
            ]
            objects = VGroup(
                *[
                    words(s, 23, FG, mono=True).move_to((x, 0.9 - t * 0.8, 0))
                    for s, x in zip(row, columns, strict=True)
                ]
            )
            self.play(FadeIn(objects), run_time=0.65)
        setup = words(
            "1 head  /  Dk = 2  /  Dv = 1  /  incoming memory = [1, 1]", 23, PURPLE
        )
        setup.move_to((0, -1.7, 0))
        self.play(FadeIn(setup))
        self.caption(
            "Keys are unit length. Queries equal keys only in this toy example.\nWe will get exactly the same outputs in two different ways.",
            8,
        )

    def follow_decay(self):
        self.chapter(3, "3. Follow a write through later gates")
        note = words(
            "A separate illustration: one correction starts as [1, 1]", 24, MUTED
        )
        note.move_to((0, 2.1, 0))
        self.play(FadeIn(note))
        bars, numbers = VGroup(), VGroup()
        for d, y in enumerate([1.1, 0.3]):
            bar = Rectangle(
                width=4,
                height=0.35,
                stroke_width=0,
                fill_color=[BLUE, GREEN][d],
                fill_opacity=0.9,
            )
            bar.move_to((-0.5, y, 0))
            bars.add(bar)
            number = words("1.00", 27, mono=True).move_to((2.5, y, 0))
            numbers.add(number)
            self.add(words(f"channel {d}", 23, MUTED).move_to((-4.1, y, 0)))
        timeline = (
            VGroup(
                *[
                    tile(s, c, 2.5)
                    for s, c in [
                        ("write at token 0", GOLD),
                        ("gate at token 1", BLUE),
                        ("gate at token 2", GREEN),
                    ]
                ]
            )
            .arrange(RIGHT, buff=0.9)
            .move_to((0, -1.0, 0))
        )
        self.play(FadeIn(bars), FadeIn(numbers), FadeIn(timeline))
        factors = VGroup(
            words("x [0.5, 0.8]", 21, BLUE, mono=True).next_to(
                timeline[1], DOWN, buff=0.18
            ),
            words("x [0.9, 0.5]", 21, GREEN, mono=True).next_to(
                timeline[2], DOWN, buff=0.18
            ),
        )
        self.play(FadeIn(factors))
        self.caption(
            "Token 0's gate acted before the correction was written.\nOnly gates at tokens 1 and 2 can decay this new correction.",
            6,
        )
        for values, step in [([0.5, 0.8], 1), ([0.45, 0.4], 2)]:
            animations = [Indicate(timeline[step], color=GOLD)]
            for d, value in enumerate(values):
                target = Rectangle(
                    width=4 * value,
                    height=0.35,
                    stroke_width=0,
                    fill_color=[BLUE, GREEN][d],
                    fill_opacity=0.9,
                )
                target.align_to(bars[d], LEFT).set_y(bars[d].get_y())
                animations += [
                    Transform(bars[d], target),
                    Transform(
                        numbers[d],
                        words(f"{value:.2f}", 27, mono=True).move_to(numbers[d]),
                    ),
                ]
            self.play(*animations, run_time=1.5)
            self.wait(2)
        formula = words("retention from i to t = exp(G_t - G_i)", 25, GOLD, mono=True)
        formula.move_to((0, -2.3, 0))
        self.play(FadeIn(formula))
        self.caption(
            "Products of retention become sums of log-retention.\nG_t is the prefix sum through token t; subtract G_i to exclude gate i.",
            7,
        )
        self.caption(
            "Incoming memory existed before the chunk.\nIts retention includes every gate from token 0 through token t: exp(G_t).",
            6,
        )

    def key_influence(self):
        self.chapter(4, "4. How much does an earlier write matter?")
        origin = (-4.5, -0.8, 0)
        horizontal = Line(origin, (-1.0, -0.8, 0), color=MUTED, stroke_width=1)
        vertical = Line(origin, (-4.5, 2.0, 0), color=MUTED, stroke_width=1)
        writer = Arrow(origin, (-1.7, -0.8, 0), buff=0, color=BLUE)
        reader = Arrow(origin, (-2.82, 1.44, 0), buff=0, color=GREEN)
        labels = VGroup(
            words("k0 = [1, 0]", 22, BLUE).move_to((-2.9, -1.3, 0)),
            words("k1 = [0.6, 0.8]", 22, GREEN).move_to((-3.1, 1.9, 0)),
        )
        self.play(
            Create(horizontal),
            Create(vertical),
            Create(writer),
            Create(reader),
            FadeIn(labels),
        )
        self.caption(
            "A key reads an earlier correction through a dot product.\nAligned directions interact strongly; orthogonal directions do not.",
            6,
        )
        decayed_writer = Arrow(origin, (-3.1, -0.8, 0), buff=0, color=BLUE)
        self.play(
            Transform(writer, decayed_writer),
            Transform(
                labels[0],
                words("decayed k0 = [0.5, 0]", 22, BLUE).move_to((-3.1, -1.3, 0)),
            ),
            run_time=1.6,
        )
        calculation = VGroup(
            words("dot(k1, decayed k0) = 0.30", 24, mono=True),
            words("writer beta0 = 0.50", 24, GOLD, mono=True),
            words("a10 = 0.50 x 0.30 = 0.15", 24, GOLD, mono=True),
        )
        calculation.arrange(DOWN, buff=0.45).move_to((2.3, 0.8, 0))
        self.play(FadeIn(calculation))
        self.caption(
            "Decay scales the writer's key channel by channel before the read.\nThe writer's beta scales how strongly that correction was written.",
            7,
        )
        impact = words("error0 = 1.20  =>  prediction1 gains 0.18", 25, GOLD, mono=True)
        impact.move_to((0, -2.2, 0))
        self.play(FadeIn(impact))
        self.caption(
            "This coefficient links error0 to token 1's prediction.\nWe can compute it from keys, gates, and beta before knowing error0.",
            6,
        )

    def dependencies(self):
        self.chapter(5, "5. Put all local dependencies in a triangle")
        nodes = VGroup(
            *[
                tile(f"error {i}", GOLD).move_to((x, 2.0, 0))
                for i, x in enumerate([-4.2, 0, 4.2])
            ]
        )
        edges = VGroup(
            Arrow(nodes[0].get_right(), nodes[1].get_left(), buff=0.12, color=GOLD),
            Arrow(nodes[1].get_right(), nodes[2].get_left(), buff=0.12, color=GOLD),
            CurvedArrow(
                nodes[0].get_bottom(), nodes[2].get_bottom(), angle=0.35, color=GOLD
            ),
        )
        matrix = grid(
            [["0", "0", "0"], [".150", "0", "0"], [".180", ".504", "0"]],
            GOLD,
            diagonal=False,
        ).move_to((-3.4, -0.65, 0))
        matrix_label = words("A: row = reader, column = writer", 21, MUTED)
        matrix_label.next_to(matrix, UP, buff=0.3)
        equations = VGroup(
            words("e0 = b0", 24, mono=True),
            words("e1 = b1 - .150 e0", 24, mono=True),
            words("e2 = b2 - .180 e0 - .504 e1", 22, mono=True),
        )
        equations.arrange(DOWN, aligned_edge=LEFT, buff=0.5).move_to((2.2, -0.5, 0))
        self.play(FadeIn(nodes))
        self.play(*[Create(e) for e in edges], run_time=1.5)
        self.play(FadeIn(matrix), FadeIn(matrix_label), FadeIn(equations))
        self.caption(
            "Errors depend only on earlier writes inside the chunk.\nA is strictly lower triangular: future writes and the current write are excluded.",
            7,
        )
        self.caption(
            "b_t is value minus the prediction from decayed incoming memory alone.\nSubtract earlier corrections to obtain the actual error e_t.",
            7,
        )

    def solve_errors(self):
        self.chapter(6, "6. Solve the same dependent equations as a block")
        system = grid(
            [["1", "0", "0"], [".150", "1", "0"], [".180", ".504", "1"]], GOLD
        ).move_to((-3.7, 0.3, 0))
        errors = grid([["?"], ["?"], ["?"]], GOLD).move_to((0, 0.3, 0))
        base = grid([[f"{v:.3f}"] for v in self.example.base_errors], PURPLE)
        base.move_to((3.5, 0.3, 0))
        labels = VGroup(
            words("I + A", 26, GOLD).next_to(system, UP, buff=0.3),
            words("E", 26, GOLD).next_to(errors, UP, buff=0.3),
            words("B", 26, PURPLE).next_to(base, UP, buff=0.3),
            words("x", 28).move_to((-1.6, 0.3, 0)),
            words("=", 30).move_to((1.65, 0.3, 0)),
        )
        self.play(FadeIn(system), FadeIn(errors), FadeIn(base), FadeIn(labels))
        self.caption(
            "Move the earlier-error terms to the left: (I + A) E = B.\nE has shape [B, H, C, Dv]; the system matrix is [B, H, C, C].",
            7,
        )
        base_note = words(
            "B = values - decayed incoming-memory predictions", 24, PURPLE
        )
        base_note.move_to((0, -2.0, 0))
        self.play(FadeIn(base_note), Indicate(base, color=PURPLE))
        self.caption(
            "Incoming memory alone predicts [0.800, 0.816, 0.504].\nSubtract these from values [2, 2, 3] to get B = [1.200, 1.184, 2.496].",
            7,
        )
        self.play(FadeOut(base_note))
        for t, explanation in enumerate(
            [
                "First row: e0 = 1.200.",
                "Second row: e1 = 1.184 - .150 x 1.200 = 1.004.",
                "Third row: e2 = 2.496 - .180 x 1.200 - .504 x 1.004 = 1.774.",
            ]
        ):
            highlight = SurroundingRectangle(
                VGroup(*system[0][t * 3 : (t + 1) * 3]), color=GOLD, buff=0.03
            )
            self.play(Create(highlight), run_time=0.4)
            target = words(f"{self.example.errors[t]:.3f}", 23, GOLD, mono=True)
            if target.width > 0.64:
                target.scale_to_fit_width(0.64)
            target.move_to(errors[1][t])
            self.play(Transform(errors[1][t], target), run_time=0.7)
            self.caption(explanation, 4)
            self.play(FadeOut(highlight), run_time=0.3)
        solver = words("E = solve_triangular(I + A, B)", 26, GOLD, mono=True)
        solver.move_to((0, -2.0, 0))
        self.play(FadeIn(solver))
        self.caption(
            "The dependencies still exist. A compiled triangular solver handles them\nand batches heads and value columns. We do not form a matrix inverse.",
            7,
        )

    def outputs_and_state(self):
        self.chapter(7, "7. Read outputs, then carry the final memory")
        old_readout = grid([[".800"], [".816"], [".504"]], PURPLE).move_to(
            (-5.4, 0.5, 0)
        )
        weights = grid(
            [[".500", "0", "0"], [".150", ".750", "0"], [".180", ".504", "1"]], GREEN
        ).move_to((-2.1, 0.5, 0))
        errors = grid([[f"{v:.3f}"] for v in self.example.errors], GOLD).move_to(
            (0.6, 0.5, 0)
        )
        outputs = grid([[f"{v:.3f}"] for v in self.example.outputs], GREEN).move_to(
            (4.3, 0.5, 0)
        )
        labels = VGroup(
            words("old readout", 21, PURPLE).next_to(old_readout, UP, buff=0.3),
            words("M", 26, GREEN).next_to(weights, UP, buff=0.3),
            words("E", 26, GOLD).next_to(errors, UP, buff=0.3),
            words("O", 26, GREEN).next_to(outputs, UP, buff=0.3),
            words("+", 29).move_to((-4.1, 0.5, 0)),
            words("x", 28).move_to((-0.5, 0.5, 0)),
            words("=", 30).move_to((2.4, 0.5, 0)),
        )
        self.play(
            FadeIn(old_readout),
            FadeIn(weights),
            FadeIn(errors),
            FadeIn(outputs),
            FadeIn(labels),
        )
        self.caption(
            "Outputs are decayed incoming-memory readouts plus M E.\nM uses queries to read writes; here queries equal keys for readable numbers.",
            7,
        )
        diag = VGroup(
            *[
                SurroundingRectangle(weights[0][i * 3 + i], color=GREEN, buff=0.01)
                for i in range(3)
            ]
        )
        self.play(*[Create(m) for m in diag])
        self.caption(
            "M includes the diagonal: each output reads its own write.\nThe error-dependency matrix A excluded the diagonal.",
            6,
        )
        final = words(
            f"S_out = [{self.example.final_state[0]:.3f}, {self.example.final_state[1]:.3f}]",
            27,
            PURPLE,
            mono=True,
        ).move_to((0, -1.7, 0))
        self.play(FadeIn(final))
        self.caption(
            "The final memory is decayed S_in plus all surviving corrections.\nThese outputs and this state match the original token loop exactly.",
            7,
        )

    def chunk_pipeline(self):
        self.chapter(8, "8. Larger work inside chunks; state between chunks")
        chunks = VGroup()
        for c, x in enumerate([-4.8, 0, 4.8]):
            block = grid(
                [["1", "0", "0"], ["a", "1", "0"], ["a", "a", "1"]], GOLD, cell=0.62
            ).move_to((x, 0.5, 0))
            title = words(f"chunk {c}", 26, BLUE).next_to(block, UP, buff=0.35)
            chunks.add(VGroup(block, title))
        arrows = VGroup(
            *[
                Arrow(
                    chunks[i][0].get_right(),
                    chunks[i + 1][0].get_left(),
                    buff=0.22,
                    color=PURPLE,
                )
                for i in range(2)
            ]
        )
        self.play(FadeIn(chunks))
        self.play(*[Indicate(c[0], color=GOLD) for c in chunks], run_time=1.2)
        self.caption(
            "Keys, queries, gates, and beta are already known for the layer input.\nBuild each chunk's local coefficients across chunks and heads in parallel.",
            7,
        )
        self.play(*[Create(a) for a in arrows])
        for i in range(2):
            state = Dot(arrows[i].get_start(), color=PURPLE, radius=0.12)
            self.add(state)
            self.play(state.animate.move_to(arrows[i].get_end()), run_time=1.5)
            self.play(Indicate(chunks[i + 1][0], color=GOLD), FadeOut(state))
        state_label = words("carry S_out -> next S_in", 25, PURPLE, mono=True)
        state_label.move_to((0, -1.4, 0))
        self.play(FadeIn(state_label))
        self.caption(
            "Our educational version still advances state from chunk to chunk.\nInside a chunk, triangular solves and matrix products replace a Python token loop.",
            7,
        )
        self.caption(
            "Fixed-size state is passed between chunks. Local token interactions are C x C.\nThe whole sequence never needs one full S x S attention matrix.",
            7,
        )

    def practical_scope(self):
        self.chapter(9, "9. What should we implement for learning?")
        rows = (
            VGroup(
                words("Build local coefficients with cumulative log-decay", 27, BLUE),
                words("Solve errors; compute outputs and outgoing state", 27, GOLD),
                words("Check every result against the recurrent reference", 27, GREEN),
            )
            .arrange(DOWN, aligned_edge=LEFT, buff=0.65)
            .move_to((0, 0.8, 0))
        )
        self.play(FadeIn(rows))
        self.caption(
            "An educational triangular-system implementation is feasible.\nThe paper's WY transforms and fused GPU kernels are a larger engineering task.",
            7,
        )
        caveat = words(
            "A naive decay broadcast may create a C x C x Dk temporary.", 23, MUTED
        )
        caveat.move_to((0, -1.6, 0))
        self.play(FadeIn(caveat))
        self.caption(
            "A block formulation exposes larger operations; it does not guarantee speed.\nNumerical stability, memory use, and the chosen backend still matter.",
            7,
        )
        self.caption(
            "Remember the idea: preserve the causal dependencies,\nbut collect them into block equations instead of individual Python updates.",
            7,
        )

    def write_captions(self):
        self.cues[-1][1] = self.time
        media = Path(config.media_dir)
        media.mkdir(parents=True, exist_ok=True)

        def timestamp(seconds):
            milliseconds = round(seconds * 1000)
            hours, remainder = divmod(milliseconds, 3_600_000)
            minutes, remainder = divmod(remainder, 60_000)
            seconds, ms = divmod(remainder, 1000)
            return f"{hours:02}:{minutes:02}:{seconds:02}.{ms:03}"

        cues = [
            f"{timestamp(start)} --> {timestamp(end)}\n{text}"
            for start, end, text in self.cues
        ]
        (media / "kda_chunking.vtt").write_text("WEBVTT\n\n" + "\n\n".join(cues) + "\n")
        for time, title in self.chapters:
            print(f"CHAPTER {timestamp(time)} {title}")
