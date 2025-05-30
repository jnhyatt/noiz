//! Contains common imports

pub use crate::{
    DynamicConfigurableSampleable, DynamicSampleable, Noise, NoiseFunction, Sampleable,
    SampleableFor, ScalableNoise, SeedableNoise,
    cell_noise::{
        BlendCellGradients, BlendCellValues, DistanceBlend, MixCellGradients, MixCellValues,
        PerCell, PerCellPointDistances, QuickGradients, SimplecticBlend, WorleyLeastDistance,
    },
    cells::{OrthoGrid, SimplexGrid, Voronoi, WithGradient},
    curves::{DoubleSmoothstep, Lerped, Linear, Smoothstep},
    layering::{
        DomainWarp, FractalLayers, LayeredNoise, Normed, NormedByDerivative, Octave,
        PeakDerivativeContribution, Persistence, SmoothDerivativeContribution,
    },
    lengths::{EuclideanLength, ManhattanLength},
    math_noise::{Billow, PingPong, SNormToUNorm, UNormToSNorm},
    misc_noise::{Masked, Offset, RandomElements, RemapCurve, Scaled, SelfMasked, Translated},
    rng::{Random, SNorm, UNorm},
};

/// Contains type aliases for common noise types.
/// This reduces some boiler plate and is educational.
pub mod common_noise {
    use super::*;

    /// A [`NoiseFunction`] that produces white noise `f32`s between 0 and 1.
    pub type White = PerCell<OrthoGrid, Random<UNorm, f32>>;

    /// A [`NoiseFunction`] that produces value noise `f32`s between 0 and 1.
    pub type Value = MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>>;

    /// A [`NoiseFunction`] that produces perlin noise `f32`s between -1 and 1.
    pub type Perlin = MixCellGradients<OrthoGrid, Smoothstep, QuickGradients>;

    /// A [`NoiseFunction`] that produces simplex noise `f32`s between -1 and 1.
    pub type Simplex = BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients>;

    /// A [`NoiseFunction`] that produces value noise `f32`s between 0 and 1 and its gradient in a [`WithGradient`].
    pub type ValueWithDerivative = MixCellValues<OrthoGrid, Smoothstep, Random<UNorm, f32>, true>;

    /// A [`NoiseFunction`] that produces perlin noise `f32`s between -1 and 1 and its gradient in a [`WithGradient`].
    pub type PerlinWithDerivative = MixCellGradients<OrthoGrid, Smoothstep, QuickGradients, true>;

    /// A [`NoiseFunction`] that produces simplex noise `f32`s between -1 and 1 and its gradient in a [`WithGradient`].
    pub type SimplexWithDerivative =
        BlendCellGradients<SimplexGrid, SimplecticBlend, QuickGradients, true>;

    /// A [`NoiseFunction`] that produces traditional worley noise `f32`s between 0 and 1.
    pub type Worley = PerCellPointDistances<Voronoi, EuclideanLength, WorleyLeastDistance>;

    /// Represents traditional fractal brownian motion.
    pub type Fbm<T> = LayeredNoise<Normed<f32>, Persistence, FractalLayers<Octave<T>>>;
}
