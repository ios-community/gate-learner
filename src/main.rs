#![deny(unsafe_code)]
#![deny(clippy::pedantic)]

//! Command-line interface for the Boole Project.

use clap::Parser;
use gate_learner::{
    cli::{Args, run_app},
    error::GateLearnerError,
};

#[cfg(feature = "dhat-heap")]
#[global_allocator]
static ALLOC: dhat::Alloc = dhat::Alloc;

fn main() -> Result<(), GateLearnerError> {
    #[cfg(feature = "dhat-heap")]
    let _profiler = dhat::Profiler::new_heap();

    let args = Args::parse();
    run_app(&args)
}
