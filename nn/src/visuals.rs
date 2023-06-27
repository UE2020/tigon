use dfdx::prelude::*;
struct SkipVisitor<'a, V>(&'a mut V);

impl<T: TensorCollection<E, D>, E: Dtype, D: Device<E>, V: ModuleVisitor<PrintOutput<T>, E, D>>
    ModuleVisitor<T, E, D> for SkipVisitor<'_, V>
{
    type Err = V::Err;
    type E2 = V::E2;
    type D2 = V::D2;

    fn visit_module<Field, GetRef, GetMut>(
        &mut self,
        name: &str,
        mut get_refs: GetRef,
        mut get_muts: GetMut,
    ) -> Result<Option<Field::To<Self::E2, Self::D2>>, Self::Err>
    where
        GetRef: FnMut(&T) -> &Field,
        GetMut: FnMut(&mut T) -> &mut Field,
        Field: TensorCollection<E, D>,
    {
        self.0
            .visit_module(name, |s| get_refs(&s.0), |s| get_muts(&mut s.0))
    }

    fn visit_tensor<S: Shape, GetRef, GetMut>(
        &mut self,
        name: &str,
        mut get_refs: GetRef,
        mut get_muts: GetMut,
        opts: TensorOptions<S, E, D>,
    ) -> Result<Option<Tensor<S, Self::E2, Self::D2>>, Self::Err>
    where
        GetRef: FnMut(&T) -> &Tensor<S, E, D>,
        GetMut: FnMut(&mut T) -> &mut Tensor<S, E, D>,
    {
        self.0
            .visit_tensor(name, |s| get_refs(&s.0), |s| get_muts(&mut s.0), opts)
    }

    fn visit_fields<M: ModuleFields<T, E, D>>(
        &mut self,
        fields: M,
        builder: impl FnOnce(M::Output<Self::E2, Self::D2>) -> T::To<Self::E2, Self::D2>,
    ) -> Result<Option<T::To<Self::E2, Self::D2>>, Self::Err> {
        let options = fields.visit_fields(self)?;
        Ok(M::handle_options(options).map(builder))
    }
}

pub struct PrintOutput<M>(M);

impl<D: Device<E>, E: Dtype, M: BuildOnDevice<D, E>> BuildOnDevice<D, E> for PrintOutput<M> {
    type Built = PrintOutput<M::Built>;
}

impl<E: Dtype, D: Device<E>, M: TensorCollection<E, D>> TensorCollection<E, D> for PrintOutput<M> {
    type To<E2: Dtype, D2: Device<E2>> = PrintOutput<M::To<E2, D2>>;

    fn iter_tensors<V: ModuleVisitor<PrintOutput<M>, E, D>>(
        visitor: &mut V,
    ) -> Result<Option<Self::To<V::E2, V::D2>>, V::Err> {
        let Some(out) = M::iter_tensors(&mut SkipVisitor(visitor))? else { return Ok(None) };
        Ok(Some(PrintOutput(out)))
    }
}

pub trait Inspect {
    fn inspect(&self);
}

impl<S: Shape, D: DeviceStorage, T> Inspect for Tensor<S, f32, D, T> {
    fn inspect(&self) {
        use image::{Rgb, RgbImage};

        let data = self.as_vec();
        let planes = ndarray::Array3::from_shape_vec((64, 8, 8), data).unwrap();
        let mut img = RgbImage::new(8 * 8, 8 * 8);
        // for x in 0..(8 * 8 + (8 - 1)) {
        // 	for y in 0..(8 * 8 + (8 - 1)) {
        // 		img.put_pixel(
        // 			x,
        // 			y,
        // 			Rgb([100, 0, 0]),
        // 		);
        // 	}
        // }
        for (i, plane) in planes.axis_iter(ndarray::Axis(0)).enumerate() {
            let mut max = 0.0;
            for x in 0..8 {
                for y in 0..8 {
                    let activation = plane[[y, x]];
                    if activation > max {
                        max = activation
                    }
                }
            }
            for x in 0..8 {
                for y in 0..8 {
                    let activation = plane[[y, x]] / max;
                    let byte = (activation * 255.0) as u8;
                    let mut color = [0, 0, 0];
                    color[i % 3] = byte;
                    // let hue = (1.0 - activation) * 240.0;
                    // let (r, g, b) = to_rgb(hue, 1.0, 0.5);
                    let x_index = i as u32 % 8;
                    let y_index = i as u32 / 8;
                    img.put_pixel(x as u32 + x_index * 8, y as u32 + y_index * 8, Rgb(color));
                }
            }
        }
        let img = image::imageops::resize(
            &img,
            8 * 8 * 5,
            8 * 8 * 5,
            image::imageops::FilterType::Triangle,
        );
        img.save("planes.png").unwrap();
    }
}

impl<I, M: Module<I>> Module<I> for PrintOutput<M>
where
    M::Output: std::fmt::Debug + Inspect,
{
    type Output = M::Output;
    type Error = M::Error;

    fn try_forward(&self, input: I) -> Result<Self::Output, Self::Error> {
        let out = self.0.try_forward(input)?;
        out.inspect();
        // let array = out.array();
        // let mut planes = ndarray::Array3::from(array);
        // println!("{:?}", out.array());
        Ok(out)
    }
}

impl<I, M: ModuleMut<I>> ModuleMut<I> for PrintOutput<M>
where
    M::Output: std::fmt::Debug,
{
    type Output = M::Output;
    type Error = M::Error;

    fn try_forward_mut(&mut self, input: I) -> Result<Self::Output, Self::Error> {
        let out = self.0.try_forward_mut(input)?;
        println!("{out:?}");
        Ok(out)
    }
}

pub fn to_rgb(h: f32, s: f32, l: f32) -> (u8, u8, u8) {
    if s == 0.0 {
        // Achromatic, i.e., grey.
        let l = percent_to_byte(l);
        return (l, l, l);
    }

    let h = h / 360.0; // treat this as 0..1 instead of degrees
    let s = s;
    let l = l;

    let q = if l < 0.5 {
        l * (1.0 + s)
    } else {
        l + s - (l * s)
    };
    let p = 2.0 * l - q;

    (
        percent_to_byte(hue_to_rgb(p, q, h + 1.0 / 3.0)),
        percent_to_byte(hue_to_rgb(p, q, h)),
        percent_to_byte(hue_to_rgb(p, q, h - 1.0 / 3.0)),
    )
}

fn percent_to_byte(percent: f32) -> u8 {
    (percent * 255.0).round() as u8
}

fn hue_to_rgb(p: f32, q: f32, t: f32) -> f32 {
    let t = if t < 0.0 {
        t + 1.0
    } else if t > 1.0 {
        t - 1.0
    } else {
        t
    };

    if t < 1.0 / 6.0 {
        p + (q - p) * 6.0 * t
    } else if t < 1.0 / 2.0 {
        q
    } else if t < 2.0 / 3.0 {
        p + (q - p) * (2.0 / 3.0 - t) * 6.0
    } else {
        p
    }
}
