import { InferenceSession, Tensor, env } from 'onnxruntime-web';
import meyda from 'meyda';
import { Component, ElementRef, OnDestroy, signal, viewChild } from '@angular/core';
import { RouterOutlet } from '@angular/router';

env.wasm.wasmPaths = '/assets/onnxruntime-web/';

@Component({
    selector: 'app-root',
    imports: [RouterOutlet],
    templateUrl: './app.component.html',
    styleUrl: './app.component.css',
})
export class AppComponent implements OnDestroy {
    audioElement = viewChild<ElementRef>('audioElement');

    title = 'AppComponent';
    isRecording = signal(false);
    transcription = '';
    mediaRecorder: MediaRecorder | null = null;
    audioChunks: Blob[] = [];
    encoderSession: InferenceSession | null = null;
    decoderSession: InferenceSession | null = null;

    async ngOnInit() {
        try {
            this.encoderSession = await InferenceSession.create('https://models.gethuman.com/onnx/encoder_model.onnx');
            this.decoderSession = await InferenceSession.create('https://models.gethuman.com/onnx/decoder_model.onnx');

            console.log('ONNX session loaded.');
        } catch (error) {
            console.error('Error loading ONNX model:', error);
        }
    }

    ngOnDestroy() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
        }
    }

    async startRecording() {
        try {
            const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
            this.mediaRecorder = new MediaRecorder(stream);
            this.mediaRecorder.ondataavailable = (event) => {
                this.audioChunks.push(event.data);
            };
            this.mediaRecorder.onstop = () => {
                this.processAudio();
            };
            this.mediaRecorder.start();
            this.isRecording.set(true);
        } catch (error) {
            console.error('Error accessing microphone:', error);
        }
    }

    stopRecording() {
        if (this.mediaRecorder) {
            this.mediaRecorder.stop();
            this.isRecording.set(false);
        }
    }

    async processAudio() {
        if (!this.decoderSession || !this.encoderSession) {
            console.error('ONNX session not loaded.');
            return;
        }

        const audioBlob = new Blob(this.audioChunks, { type: 'audio/wav' }); // Ensure wav format or adapt preprocessing
        const audioUrl = URL.createObjectURL(audioBlob);
        const audio = this.audioElement();
        if (audio) {
            audio.nativeElement.src = audioUrl;
        }

        const audioBuffer = await audioBlob.arrayBuffer();
        // Preprocess the audio buffer to the format expected by the Whisper model.
        // This is the most complex part and requires careful attention to the model's input requirements.
        const processedAudio = await this.preprocessAudio(audioBuffer);

        if (!processedAudio) {
            console.error('Audio preprocessing failed');
            return;
        }

        try {
            const encoderInput = new Tensor('float32', processedAudio, [1, processedAudio.length, 1]); // Adjust shape as needed
            const encoderFeeds = { input_features: encoderInput };
            const encoderResults = await this.encoderSession.run(encoderFeeds);

            const decoderInput = encoderResults['output']; // encoder output is decoder input.
            const decoderFeeds = { input_features: decoderInput };
            const decoderResults = await this.decoderSession.run(decoderFeeds);

            // Postprocess the output of the Whisper model to get the transcribed text.
            this.transcription = await this.postprocessOutput(decoderResults['output']);
        } catch (error) {
            console.error('Error running ONNX model:', error);
        } finally {
            this.audioChunks = []; // Clear chunks for next recording
        }
    }

    async preprocessAudio(audioBuffer: ArrayBuffer): Promise<Float32Array | null> {
        // Implement audio preprocessing here. This typically involves:
        // 1. Decoding the audio buffer (e.g., using Web Audio API or a library like ffmpeg.wasm).
        // 2. Resampling the audio to the model's expected sample rate (usually 16kHz).
        // 3. Converting the audio to mono.
        // 4. Optionally, applying other transformations like normalization or feature extraction (e.g., MFCCs).
        // 5. Returning the processed audio as a Float32Array.
        // placeholder code.  Replace with actual audio processing.
        try {
            const audioContext = new AudioContext({ sampleRate: 16000 }); // important to match whisper model
            const decodedAudio = await audioContext.decodeAudioData(audioBuffer);
            const audioData = decodedAudio.getChannelData(0); // mono

            // TODO: need to implement log-Mel spectrogram extraction
            // TODO: createcontext tokens (very complex without transformers)

            return audioData;
        } catch (e) {
            console.error('Error decoding audio', e);
            return null;
        }

        // try {
        //     const audioContext = new AudioContext({ sampleRate: 16000 });
        //     const decodedAudio = await audioContext.decodeAudioData(audioBuffer);
        //     const audioData = decodedAudio.getChannelData(0); // mono

        //     // Example MFCC extraction (adjust parameters as needed)
        //     const mfccs = meyda.extract('mfcc', audioData, {
        //         sampleRate: 16000,
        //         bufferSize: 512, // window size
        //         hopSize: 256, // stride
        //         numberOfMFCCCoefficients: 80, // Number of MFCCs to match model.
        //     });

        //     // const analyzer = meyda.createMeydaAnalyzer({
        //     //     audioContext: audioContext,
        //     //     source: audioBuffer,
        //     //     bufferSize: 512,
        //     //     hopSize: 256,
        //     //     featureExtractors: ['rms'],
        //     //     // callback: (features) => {
        //     //     //     console.log(features);
        //     //     // },
        //     // });
        //     // analyzer.start();

        //     if (!Array.isArray(mfccs)) {
        //         console.error('meyda did not return an array');
        //         return null;
        //     }

        //     return Float32Array.from(mfccs);
        // } catch (e) {
        //     console.error('Error decoding audio or computing MFCCs', e);
        //     return null;
        // }
    }

    async postprocessOutput(outputTensor: Tensor): Promise<string> {
        // Implement postprocessing here. This typically involves:
        // 1. Extracting the predicted tokens from the output tensor.
        // 2. Decoding the tokens into text (e.g., using a tokenizer or vocabulary).
        // This part is highly dependent on the specific Whisper model you are using.
        // Placeholder - replace with actual postprocessing
        const outputData = outputTensor.data;
        if (outputData instanceof Float32Array) {
            // Very basic example. A real implementation is much more complex.
            // This will not produce usable output and is only for demonstration purposes.
            return Array.from(outputData)
                .map((x) => String.fromCharCode(Math.round(x)))
                .join('');
        } else {
            return 'Postprocessing error.';
        }
    }
}
