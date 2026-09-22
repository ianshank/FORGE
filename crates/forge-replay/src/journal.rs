use std::fs::{File, OpenOptions};
use std::io::{BufWriter, Write};
use std::path::Path;

use forge_core::events::SimulationEvent;
use serde::{Deserialize, Serialize};
use tracing::{error, instrument};

/// A journal entry representing a deterministic state transition or event.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum JournalEntry {
    /// A discrete simulation event.
    Event(SimulationEvent),
    /// A raw action proposal from an agent or external controller.
    ActionProposal {
        /// Agent or controller making the proposal.
        agent_id: u32,
        /// Type of action.
        action_type: String,
        /// Serialized action payload.
        payload: Vec<u8>,
        /// Tick when proposal was made.
        tick: u64,
    },
    /// A tick boundary marker indicating the end of a simulation step.
    TickBoundary(u64),
}

/// An authoritative append-only journal for recording all simulation events
/// and state transitions before they are exported to external memory mirrors
/// like Honcho or consumed by DeerFlow.
pub struct AppendOnlyJournal {
    writer: BufWriter<File>,
    /// Number of bytes written to the journal.
    bytes_written: u64,
}

impl AppendOnlyJournal {
    /// Creates or opens an append-only journal at the specified path.
    #[instrument(level = "debug", skip(path))]
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(path)?;

        Ok(Self {
            writer: BufWriter::new(file),
            bytes_written: 0,
        })
    }

    /// Appends a journal entry to the log.
    ///
    /// This method enforces the "append-only" integrity property by immediately
    /// serializing the event via bincode and writing it to disk.
    #[instrument(level = "trace", skip(self, entry))]
    pub fn append(&mut self, entry: &JournalEntry) -> std::io::Result<()> {
        let serialized = match bincode::serialize(entry) {
            Ok(bytes) => bytes,
            Err(e) => {
                error!(error = %e, "failed to serialize journal entry");
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "serialization error",
                ));
            }
        };

        // Write length prefix (4 bytes) followed by the payload
        let len = serialized.len() as u32;
        self.writer.write_all(&len.to_le_bytes())?;
        self.writer.write_all(&serialized)?;
        self.bytes_written += 4 + len as u64;

        Ok(())
    }

    /// Flushes the underlying writer to disk.
    pub fn flush(&mut self) -> std::io::Result<()> {
        self.writer.flush()
    }

    /// Returns the total bytes written in this session.
    pub fn bytes_written(&self) -> u64 {
        self.bytes_written
    }
}

/// A reader for the authoritative append-only journal.
pub struct AppendOnlyJournalReader {
    reader: std::io::BufReader<File>,
}

impl AppendOnlyJournalReader {
    /// Opens a journal for reading.
    pub fn new<P: AsRef<Path>>(path: P) -> std::io::Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            reader: std::io::BufReader::new(file),
        })
    }

    /// Reads the next entry from the journal. Returns `Ok(None)` on EOF.
    pub fn next_entry(&mut self) -> std::io::Result<Option<JournalEntry>> {
        use std::io::Read;

        let mut len_bytes = [0u8; 4];
        match self.reader.read_exact(&mut len_bytes) {
            Ok(_) => {}
            Err(e) if e.kind() == std::io::ErrorKind::UnexpectedEof => return Ok(None),
            Err(e) => return Err(e),
        }

        let len = u32::from_le_bytes(len_bytes) as usize;
        let mut payload = vec![0u8; len];
        self.reader.read_exact(&mut payload)?;

        match bincode::deserialize(&payload) {
            Ok(entry) => Ok(Some(entry)),
            Err(e) => {
                error!(error = %e, "failed to deserialize journal entry");
                Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidData,
                    "deserialization error",
                ))
            }
        }
    }
}
