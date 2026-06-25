#![deny(unsafe_code)]
#![deny(clippy::pedantic)]

//! Command-line interface for the Boole Project.

use clap::Parser;
use gate_learner::{
    cli::{Args, run_app},
    error::GateLearnerError,
};

fn main() -> Result<(), GateLearnerError> {
    let args = Args::parse();
    run_app(&args)
}
