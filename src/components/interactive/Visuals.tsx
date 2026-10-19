/**
 * Brief 41's chapter heroes with their equations, for the chapters' MDX.
 * Server Components: KaTeX renders here, the animations load in the
 * browser. Highlight keys: the `hl-*` classes in globals.css.
 */
import { Eq } from "@/components/mdx/Eq";

import { HeadSharingWidget, MoeRoutingWidget, RopePhasorWidget } from "./lazy";

/** Chapter 1: the KV cache per token is layers × units × numbers × bytes. */
export function HeadSharing(): JSX.Element {
  return (
    <HeadSharingWidget
      equation={
        <Eq
          label="KV cache bytes per token: the number of layers, times the key and value units each layer caches, times the numbers in each unit, times the bytes per number. For grouped attention the units are the KV heads and each holds a key and a value of the head width; for MLA one latent."
          tex={String.raw`\frac{\text{KV bytes}}{\text{token}} = L \cdot \htmlClass{hl-a}{n_{\text{kv}}} \cdot \htmlClass{hl-w}{e_{\text{unit}}} \cdot b, \qquad e_{\text{unit}} = 2\,d_h \ \ (\text{MHA, GQA, MQA}), \qquad e_{\text{unit}} = d_c + d_h^{R} \ \ (\text{MLA})`}
        />
      }
    />
  );
}

/** Chapter 2: rotation by position; the score depends on m − n. */
export function RopePhasors(): JSX.Element {
  return (
    <RopePhasorWidget
      equation={
        <Eq
          label="Pair i of the query at position m is rotated by m times omega i, with omega i equal to theta to the minus 2 i over d. The dot product of a rotated query and key is a sum over pairs of the product of their lengths times the cosine of (m minus n) omega i plus their phase difference: it depends only on m minus n."
          tex={String.raw`\begin{gathered} \htmlClass{hl-a}{q^{(i)}_m = R(m\,\omega_i)\,q^{(i)}}, \quad \omega_i = \theta^{-2i/d} \\ \htmlClass{hl-p}{q_m \cdot k_n = \textstyle\sum_i \lvert q^{(i)} \rvert\, \lvert k^{(i)} \rvert \cos\!\big((m-n)\,\omega_i + \phi_i\big)} \end{gathered}`}
        />
      }
    />
  );
}

/** Chapter 4: top-k by affinity plus bias; the bias follows the load. */
export function MoeRouting(): JSX.Element {
  return (
    <MoeRoutingWidget
      equation={
        <Eq
          label="Affinity s of token t for expert i is the sigmoid of the token's vector dotted with the expert's centroid. Each token keeps the top k experts by affinity plus the expert's bias, and weights them by their affinities alone. After a step, each expert's bias drops by gamma if its load c was over the mean and rises by gamma if under."
          tex={String.raw`s_{i,t} = \sigma\!\big(u_t^{\top} e_i\big), \qquad \htmlClass{hl-c}{\text{keep top-}k \text{ of } s_{i,t} + b_i}, \qquad g_{i,t} = \frac{s_{i,t}}{\sum_{j \in \text{kept}} s_{j,t}}, \qquad \htmlClass{hl-u}{b_i \leftarrow b_i - \gamma \operatorname{sign}(c_i - \bar c)}`}
        />
      }
    />
  );
}
