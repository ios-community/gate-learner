#![deny(unsafe_code)]
#![deny(missing_docs)]
#![deny(rustdoc::broken_intra_doc_links)]
#![deny(rustdoc::private_intra_doc_links)]
#![deny(rustdoc::missing_crate_level_docs)]
#![deny(rustdoc::invalid_codeblock_attributes)]
#![deny(rustdoc::invalid_html_tags)]
#![deny(rustdoc::invalid_rust_codeblocks)]
#![deny(rustdoc::bare_urls)]
#![deny(clippy::pedantic)]
#![allow(clippy::module_name_repetitions)]
#![allow(clippy::cast_precision_loss)]
#![allow(clippy::cast_possible_wrap)]
#![allow(clippy::cast_possible_truncation)]
#![allow(clippy::similar_names)]

//! # Gate Learner Library
//!
//! This library provides a modern multilayer perceptron (MLP) implementation built from scratch,
//! dataset generation for $N$-input logic gates (OR and XOR), model and history serialisation,
//! and optional training history plotting.

pub mod cli;
pub mod core;
pub mod dataset;
pub mod error;
pub mod plot;
pub mod storage;
