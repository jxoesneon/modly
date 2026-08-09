import { useAppStore } from '@shared/stores/appStore'
import { Section, Card, Row, Select } from '@shared/ui'

export function PerformanceSection(): JSX.Element {
  const { device, precision, setDevice, setPrecision } = useAppStore()

  return (
    <Section title="Performance" subtitle="Configure GPU usage and memory limits.">
      <div className="grid grid-cols-2 gap-4">

        <Card title="Device" description="Select which compute device to use for AI inference.">
          <Row label="GPU device" description="Device used for model inference.">
            <Select value={device} onChange={(v) => setDevice(v as any)} options={[
              { value: 'auto',  label: 'Auto-detect' },
              { value: 'mps',   label: 'Apple GPU (MPS)' },
              { value: 'cuda0', label: 'NVIDIA GPU 0' },
              { value: 'cuda1', label: 'NVIDIA GPU 1' },
              { value: 'cpu',   label: 'CPU (slow)' },
            ]} />
          </Row>
          <Row label="Precision" description="Floating-point precision for inference. BF16 requires M2+.">
            <Select value={precision} onChange={(v) => setPrecision(v as any)} options={[
              { value: 'auto', label: 'Auto (FP16 on GPU, FP32 on CPU)' },
              { value: 'fp16', label: 'FP16 (half precision)' },
              { value: 'bf16', label: 'BF16 (bfloat16, M2+)' },
              { value: 'fp32', label: 'FP32 (full precision)' },
            ]} />
          </Row>
        </Card>

        <Card title="Memory" description="Control memory allocation per generation job.">
          <Row label="VRAM limit" description="Max GPU memory per generation.">
            <Select value={'8'} onChange={() => {}} options={[
              { value: '4',  label: '4 GB' },
              { value: '6',  label: '6 GB' },
              { value: '8',  label: '8 GB' },
              { value: '12', label: '12 GB' },
              { value: '0',  label: 'No limit' },
            ]} />
          </Row>
          <Row label="Parallel workers" description="Concurrent generation jobs.">
            <Select value={'1'} onChange={() => {}} options={[
              { value: '1', label: '1 (default)' },
              { value: '2', label: '2' },
              { value: '4', label: '4' },
            ]} />
          </Row>
        </Card>

      </div>
    </Section>
  )
}
