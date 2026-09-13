fn main() {
    println!("cargo:rerun-if-changed=third_party/soxr");
    let dst = cmake::Config::new("third_party/soxr")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("WITH_OPENMP", "OFF")
        .define("WITH_LSR_BINDINGS", "OFF")
        .define("WITH_DEV_TRACE", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .profile("Release")
        .build();
    let lib = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=soxr");
    // soxr's CMake links libm privately; rustc does not inherit that.
    if std::env::var("CARGO_CFG_TARGET_ENV").ok().as_deref() != Some("msvc") {
        println!("cargo:rustc-link-lib=m");
    }
}
