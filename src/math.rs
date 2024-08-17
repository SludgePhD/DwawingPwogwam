use std::{
    array,
    ops::{Add, AddAssign, Div, Mul, RangeInclusive, Sub},
};

use bytemuck::NoUninit;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(transparent)]
pub struct Vec<T, const N: usize>([T; N]);

impl<const N: usize> Vec<f32, N> {
    pub fn dist(self, other: Self) -> f32 {
        let mut sum = 0.0;
        for (&a, &b) in self.0.iter().zip(&other.0) {
            let diff = b - a;
            sum += diff * diff;
        }
        sum.sqrt()
    }

    pub fn length(self) -> f32 {
        self.dist(Vec([0.0; N]))
    }

    pub fn normalize(self) -> Self {
        self / self.length()
    }
}

// Safety: `[T; N]` has no padding iff `T` has no padding.
unsafe impl<T: NoUninit, const N: usize> NoUninit for Vec<T, N> {}

pub type Vec2<T> = Vec<T, 2>;
pub type Vec2f = Vec2<f32>;
pub type Vec2u = Vec2<u32>;

impl<T, const N: usize> From<[T; N]> for Vec<T, N> {
    fn from(value: [T; N]) -> Self {
        Self(value)
    }
}

impl<T, const N: usize> From<Vec<T, N>> for [T; N] {
    fn from(value: Vec<T, N>) -> Self {
        value.0
    }
}

impl<T, const N: usize> Add<Vec<T, N>> for Vec<T, N>
where
    T: Add<Output = T> + Copy,
{
    type Output = Vec<T, N>;

    fn add(self, rhs: Vec<T, N>) -> Self::Output {
        Vec(array::from_fn(|i| self.0[i] + rhs.0[i]))
    }
}

impl<T, const N: usize> AddAssign<Vec<T, N>> for Vec<T, N>
where
    T: Add<Output = T> + Copy,
{
    fn add_assign(&mut self, rhs: Vec<T, N>) {
        *self = *self + rhs;
    }
}

impl<T, const N: usize> Sub<Vec<T, N>> for Vec<T, N>
where
    T: Sub<Output = T> + Copy,
{
    type Output = Vec<T, N>;

    fn sub(self, rhs: Vec<T, N>) -> Self::Output {
        Vec(array::from_fn(|i| self.0[i] - rhs.0[i]))
    }
}

impl<T, const N: usize> Mul<Vec<T, N>> for Vec<T, N>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Vec<T, N>;

    fn mul(self, rhs: Vec<T, N>) -> Self::Output {
        Vec(array::from_fn(|i| self.0[i] * rhs.0[i]))
    }
}

impl<T, const N: usize> Mul<T> for Vec<T, N>
where
    T: Mul<Output = T> + Copy,
{
    type Output = Vec<T, N>;

    fn mul(self, rhs: T) -> Self::Output {
        Vec(array::from_fn(|i| self.0[i] * rhs))
    }
}

impl<T, const N: usize> Div<T> for Vec<T, N>
where
    T: Div<Output = T> + Copy,
{
    type Output = Vec<T, N>;

    fn div(self, rhs: T) -> Self::Output {
        Vec(array::from_fn(|i| self.0[i] / rhs))
    }
}

pub fn vec2<T>(x: T, y: T) -> Vec2<T> {
    Vec([x, y])
}

pub fn lerp<R>(range: RangeInclusive<R>, t: f32) -> R
where
    R: Copy + Add<R, Output = R> + Mul<f32, Output = R>,
{
    *range.start() * (1.0 - t) + *range.end() * t
}
