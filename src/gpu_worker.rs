use chan;
use futures::sync::mpsc;
use futures::{Future, Sink};
use miner::{Buffer, NonceData};
use ocl::GpuContext;
use ocl::{gpu_hash, gpu_transfer};
use reader::ReadReply;
use std::sync::Arc;

pub fn create_gpu_worker_task(
    benchmark: bool,
    rx_read_replies: chan::Receiver<ReadReply>,
    tx_empty_buffers: chan::Sender<Box<Buffer + Send>>,
    tx_nonce_data: mpsc::Sender<NonceData>,
    context_mu: Arc<GpuContext>,
) -> impl FnOnce() {
    move || {
        for read_reply in rx_read_replies {
            let mut buffer = read_reply.buffer;

            if read_reply.info.len == 0 || benchmark {
                tx_empty_buffers.send(buffer).unwrap();
                continue;
            }

            gpu_transfer(
                context_mu.clone(),
                buffer.get_gpu_buffers().unwrap(),
                *read_reply.info.gensig,
            );
            let result = gpu_hash(
                context_mu.clone(),
                read_reply.info.len / 64,
                buffer.get_gpu_data().as_ref().unwrap(),
                true,
            );
            let deadline = result.0;
            let offset = result.1;

            tx_nonce_data
                .clone()
                .send(NonceData {
                    height: read_reply.info.height,
                    deadline,
                    nonce: offset + read_reply.info.start_nonce,
                    reader_task_processed: read_reply.info.finished,
                    account_id: read_reply.info.account_id,
                })
                .wait()
                .expect("failed to send nonce data");

            tx_empty_buffers.send(buffer).unwrap();
        }
    }
}
