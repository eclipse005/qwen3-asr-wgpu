fn main() {
    println!("cargo:rerun-if-changed=third_party/soxr");
    let msvc = std::env::var("CARGO_CFG_TARGET_ENV").ok().as_deref() == Some("msvc");
    let is_apple_silicon = std::env::var("CARGO_CFG_TARGET_ARCH").ok().as_deref() == Some("aarch64")
        && std::env::var("CARGO_CFG_TARGET_OS").ok().as_deref() == Some("macos");
    
    let opt = if msvc { "/O2" } else { "-O3" };  // Use -O3 for better optimization
    
    let mut config = cmake::Config::new("third_party/soxr");
    config
        .define("BUILD_SHARED_LIBS", "OFF")
        .define("BUILD_TESTS", "OFF")
        .define("BUILD_EXAMPLES", "OFF")
        .define("WITH_OPENMP", "OFF")
        .define("WITH_LSR_BINDINGS", "OFF")
        .define("WITH_DEV_TRACE", "OFF")
        .define("CMAKE_POSITION_INDEPENDENT_CODE", "ON")
        .cflag(opt)
        .profile("Release");
    
    // Apple Silicon specific optimizations
    if is_apple_silicon {
        config
            .cflag("-mcpu=apple-m1")  // Optimize for Apple Silicon
            .cflag("-mtune=native");
        println!("cargo:rustc-env=MACOSX_DEPLOYMENT_TARGET=11.0");
    }
    
    let dst = config.build();
    let lib = dst.join("lib");
    println!("cargo:rustc-link-search=native={}", lib.display());
    println!("cargo:rustc-link-lib=static=soxr");
    if !msvc {
        println!("cargo:rustc-link-lib=m");
    }
    
    // Link Accelerate framework on macOS for optimized BLAS
    if is_apple_silicon {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
