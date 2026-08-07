use serde::{Deserialize, Serialize};
use std::{
    env,
    fs::{create_dir_all, read_to_string},
    io::{self, Write},
    path::PathBuf,
};
use tempfile::NamedTempFile;

// The program state persisted by the run subcommand.
#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct State {}

// Locate the platform-specific file where the program state is persisted.
fn path() -> Option<PathBuf> {
    // [tag:state_path_has_parent]
    dirs::data_local_dir()
        .or_else(|| {
            // Fall back to `LOCALAPPDATA` when the platform data directory is unavailable.
            env::var("LOCALAPPDATA").ok().map(Into::into)
        })
        .map(|path| path.join("stockholm/state.yml"))
}

// Return the state in which the program starts if no state was loaded from disk.
pub fn initial() -> State {
    State {}
}

// Load the program state from disk.
pub fn load() -> io::Result<State> {
    // Read and deserialize the state when a platform data directory is available.
    if let Some(path) = path() {
        let yaml = read_to_string(path)?;
        yaml_serde::from_str(&yaml).map_err(io::Error::other)
    } else {
        Err(io::Error::other("Unable to locate data directory."))
    }
}

// Save the program state to disk atomically.
pub fn save(state: &State) -> io::Result<()> {
    // Serialize and persist the state when a platform data directory is available.
    if let Some(path) = path() {
        // The `unwrap` is safe due to [ref:state_path_has_parent].
        let parent = path.parent().unwrap().to_owned();

        // The `unwrap` is safe because serialization should never fail.
        let payload = yaml_serde::to_string(state).unwrap();

        // Create the ancestor directories before atomically replacing the state file.
        create_dir_all(parent.clone())?;
        let mut temp_file = NamedTempFile::new_in(parent)?;
        temp_file.write_all(payload.as_bytes())?;
        temp_file.flush()?;
        temp_file.persist(path)?;
    } else {
        return Err(io::Error::other("Unable to locate data directory."));
    }

    Ok(())
}
