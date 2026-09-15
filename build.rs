fn main() {
    println!("cargo:rerun-if-changed=third_party/soxr");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").ok().as_deref() == Some("msvc");
    let opt = if msvc { "/O2" } else { "-O2" };
    let dst = cmake::Config::new("third_party/soxr")
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("WITH_OPENMP", "OFF")
        .define("WITH_LSR_BINDINGS", "OFF")
        .define("WITH_DEV_TRACE", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .cflag(opt)
        .profile("Release")
        .build();
    let lib = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=soxr");
    if !msvc {
        println!("cargo:rustc-link-lib=m");
    }
}
