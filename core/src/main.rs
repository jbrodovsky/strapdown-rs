use strapdown::earth;
fn main() {

    let mag = earth::calculate_magnetic_field(&90.0, &0.0, &0.0);
    println!("Mag vector: [{:.2}, {:.2}, {:.2}]", mag[0], mag[1], mag[2]);
}
