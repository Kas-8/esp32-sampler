/*
  ESP32-S3 Audio Sampler - PSRAM Optimized Final Version
  For Official Espressif ESP32-S3-DevKitC-1-N16R8 with 8MB PSRAM
  
  Hardware Configuration:
  - INMP441 mic: SD=GPIO6, SCK=GPIO5, WS=GPIO4
  - PCM5102A DAC: BCK=GPIO5, DIN=GPIO7, LCK=GPIO4
  - OLED: SDA=GPIO8, SCL=GPIO9
  - SD Card: MISO=GPIO13, SCK=GPIO12, MOSI=GPIO11, CS=GPIO10
  - Rotary Encoder: CLK=GPIO38, DT=GPIO39, SW=GPIO40, GND=GND, +=3.3V
  - Record Button: GPIO42
  - Record LED: GPIO21
  - MIDI Input: GPIO3 (RXD0)
*/

#include <Arduino.h>
#include <SD.h>
#include <SPI.h>
#include <Adafruit_SSD1306.h>
#include <vector>
#include <driver/i2s.h>
#include <Wire.h>
#include <math.h>
#include <esp_heap_caps.h>

// ------------------- ESP32-S3 Pin Configuration -------------------
constexpr int BUTTON_PIN   = 42;   // Record button
constexpr int LED_PIN      = 21;   // Record status LED
constexpr int SD_CS        = 10;   // SD Card CS
constexpr int ROTARY_CLK   = 38;   // Rotary encoder CLK
constexpr int ROTARY_DT    = 39;   // Rotary encoder DT
constexpr int ROTARY_SW    = 40;   // Rotary encoder switch
constexpr int OLED_SDA     = 8;    // OLED SDA
constexpr int OLED_SCL     = 9;    // OLED SCL
constexpr int OLED_RESET   = -1;

constexpr int I2S_WS       = 4;    // Shared by DAC LCK and MIC WS
constexpr int I2S_SCK      = 5;    // Shared by DAC BCK and MIC SCK
constexpr int I2S_MIC_SD   = 6;    // INMP441 Data Out
constexpr int I2S_DAC_DOUT = 7;    // PCM5102A Data In

constexpr int MIDI_RX_PIN  = 3;    // MIDI input via 6N137

// ------------------- PSRAM-Optimized Audio Configuration -------------------
constexpr int      SAMPLE_RATE           = 44100;
constexpr uint16_t WAV_BITS_PER_SAMPLE   = 16;
constexpr i2s_bits_per_sample_t RECORD_RX_BITS = I2S_BITS_PER_SAMPLE_32BIT;
constexpr i2s_bits_per_sample_t PLAY_TX_BITS   = I2S_BITS_PER_SAMPLE_32BIT;
constexpr unsigned long MAX_RECORD_TIME_MS = 10000; // NEW: 10-second record limit

// With 8MB PSRAM, we can use large buffers for smooth performance
constexpr size_t   RECORD_BUFFER_SIZE    = 16384;  // 16KB record buffer
constexpr int      WAV_HEADER_SIZE       = 44;
constexpr float    PREVIEW_PLAYBACK_GAIN = 1.25f;

// Mixer configuration - optimized for 8MB PSRAM
constexpr int MIXER_FRAMES_PER_CHUNK = 1024;  // Large chunks for efficiency
constexpr int NUM_VOICES             = 8;     // Full 8-voice polyphony
constexpr int MIDI_ROOT_NOTE         = 60;    // C4

// Large voice buffers in PSRAM eliminate glitching
constexpr size_t VOICE_BUFFER_SAMPLES = 32768;  // 64KB per voice buffer (32768 * 2 bytes)
constexpr size_t VOICE_BUFFER_BYTES = VOICE_BUFFER_SAMPLES * sizeof(int16_t);
constexpr size_t BASE_REFILL_THRESHOLD = 8192;   // Refill when 8KB remains
constexpr size_t MIN_REFILL_THRESHOLD = 4096;   
constexpr size_t MAX_REFILL_THRESHOLD = 16384;

// PSRAM detection
static bool psram_available = false;

// ------------------- OLED Display -------------------
constexpr int SCREEN_WIDTH  = 128;
constexpr int SCREEN_HEIGHT = 64;
TwoWire I2C_BUS = TwoWire(0);
Adafruit_SSD1306 display(SCREEN_WIDTH, SCREEN_HEIGHT, &I2C_BUS, OLED_RESET);

// ------------------- UI State -------------------
enum class UIScreen { SAMPLER, BROWSER };
UIScreen currentScreen = UIScreen::SAMPLER;
std::vector<String> fileList;
int selectedIndex = 0, scrollOffset = 0;
bool showSideMenu = false, needsDisplayUpdate = true;
constexpr int maxVisibleItems = 4;
enum class MenuState : uint8_t { Ready = 0, WaitRelease = 1, Accept = 2 };
MenuState menuState = MenuState::Ready;
int sideMenuIndex = 0, sideMenuScrollOffset = 0;
constexpr int maxSideMenuItems = 5;
int samplerFocusIndex = 0;
String assignedSampleName = "No Sample";
bool assignedNameDirty = true;
uint16_t assignedNameW = 0, assignedNameH = 0;

// ------------------- SD Card & Mutex -------------------
SemaphoreHandle_t sdMutex;
bool sd_ok = false;
unsigned long lastSDRetry = 0;
uint32_t fileIndex = 1;
struct SDGuard {
  SemaphoreHandle_t m;
  explicit SDGuard(SemaphoreHandle_t mtx) : m(mtx) { xSemaphoreTake(m, portMAX_DELAY); }
  ~SDGuard() { xSemaphoreGive(m); }
};

// ------------------- App State -------------------
enum class AppState { IDLE, RECORDING, COOLING, BLINKING, PLAYING };
AppState state = AppState::IDLE;
unsigned long stateStart = 0;
TaskHandle_t mixerTaskHandle = NULL;
TaskHandle_t playbackTaskHandle = NULL;
unsigned long lastEncoderPoll = 0;
bool lastRotaryBtn = HIGH, lastClk = HIGH;

// Task synchronization
SemaphoreHandle_t mixerStopSemaphore;
volatile bool mixerTaskStopped = false;
volatile bool requestMixerStop = false;

// ------------------- Recording -------------------
File recordingFile;
uint8_t *rxBuffer = nullptr;      // Allocated in PSRAM
uint8_t *wavOutBuffer = nullptr;  // Allocated in PSRAM
unsigned long totalBytesWritten = 0;
bool wasRecordPressed = false;
unsigned long lastRecordDebounce = 0;
constexpr unsigned long recordDebounceDelay = 30;

// ------------------- Preview Playback -------------------
volatile bool isPlaying = false;
String playbackFilename;
static const size_t PREVIEW_READ_BUF = 32768;  // 32KB preview buffer
static const int    PREVIEW_FRAMES_PER_CHUNK = 2048;
uint8_t  *preview_rb = nullptr;      // Allocated in PSRAM
int32_t  *preview_stereo32 = nullptr;  // Allocated in PSRAM
unsigned long blinkStart = 0;
int blinkState = 0, blinkCount = 0;

// ------------------- Poly Sampler -------------------
volatile bool mixerEnabled = true;
float gNoteRatio[128];

enum class VoiceState { Inactive, Playing, Finished };

typedef struct {
  volatile VoiceState state;
  uint8_t   note;
  float     phase;
  float     phaseInc;
  uint32_t  startedAtMs;
  File      file;
  int16_t   *buffer;  // Allocated in PSRAM
  size_t    buffer_len;
  volatile bool needs_refill;
  bool      is_eof;
  bool      file_valid;
  size_t    refill_threshold;
  float     last_phase_inc;
} Voice;
Voice voices[NUM_VOICES];
portMUX_TYPE voicesMux = portMUX_INITIALIZER_UNLOCKED;

// MIDI Message Queue
struct MidiMessage {
  bool      isNoteOn;
  uint8_t   note;
  uint8_t   velocity;
};
QueueHandle_t midiQueue;

// ====== Function Prototypes ======
void trySDInit();
void updateDisplay();
void drawSamplerScreen();
void drawSamplesBrowser();
void pollRotaryAndButton();
void playbackTask(void *param);
void mixerTask(void *param);
void i2s_uninstall_safe();
void i2s_apply_profile(bool tx, bool stereo);
void showStartupMenu();
void drawTextCentered(const String &txt, int16_t cx, int16_t cy, uint8_t size);
void resetVoicesAndParser();
void refillVoiceBuffers();
void processMidiQueue();
void voiceNoteOn(uint8_t note, uint8_t vel);
void voiceNoteOff(uint8_t note);
void triggerVoiceOn(uint8_t note, uint8_t vel);
void triggerVoiceOff(uint8_t note);
void cleanupFinishedVoices();
bool stopMixerTaskSafely();
void startMixerTaskSafely();
size_t calculateRefillThreshold(float phaseInc);
bool allocatePSRAMBuffers();
void recordAudioLoop();
void handleRecordingButton();
bool startRecording();
void finishRecording();
void patchWavHeader(File &file, uint32_t totalBytes);
void handleMIDI();
void readFilesIntoList();
void updateFileIndexOnBoot();
String getNextFilename();

// ====== PSRAM Memory Allocation ======
bool allocatePSRAMBuffers() {
  psram_available = (ESP.getPsramSize() > 0);
  
  if (psram_available) {
    Serial.printf("[PSRAM] Detected %d bytes (%.2f MB)\n", 
                  ESP.getPsramSize(), ESP.getPsramSize() / (1024.0 * 1024.0));
    Serial.printf("[PSRAM] Free: %d bytes\n", ESP.getFreePsram());
    
    // Allocate large voice buffers in PSRAM
    for (int v = 0; v < NUM_VOICES; v++) {
      voices[v].buffer = (int16_t*)heap_caps_malloc(VOICE_BUFFER_BYTES, MALLOC_CAP_SPIRAM);
      if (!voices[v].buffer) {
        Serial.printf("[PSRAM] Failed to allocate voice buffer %d\n", v);
        return false;
      }
    }
    
    // Allocate recording buffers in PSRAM
    rxBuffer = (uint8_t*)heap_caps_malloc(RECORD_BUFFER_SIZE, MALLOC_CAP_SPIRAM);
    wavOutBuffer = (uint8_t*)heap_caps_malloc(RECORD_BUFFER_SIZE / 2, MALLOC_CAP_SPIRAM);
    
    // Allocate preview buffers in PSRAM
    preview_rb = (uint8_t*)heap_caps_malloc(PREVIEW_READ_BUF, MALLOC_CAP_SPIRAM);
    preview_stereo32 = (int32_t*)heap_caps_malloc(PREVIEW_FRAMES_PER_CHUNK * 2 * sizeof(int32_t), MALLOC_CAP_SPIRAM);
    
    if (!rxBuffer || !wavOutBuffer || !preview_rb || !preview_stereo32) {
      Serial.println("[PSRAM] Failed to allocate work buffers");
      return false;
    }
    
    Serial.println("[PSRAM] All buffers allocated successfully");
    Serial.printf("[PSRAM] Total allocated: ~%d KB for audio buffers\n", 
                  (NUM_VOICES * VOICE_BUFFER_BYTES + RECORD_BUFFER_SIZE * 1.5 + 
                   PREVIEW_READ_BUF + PREVIEW_FRAMES_PER_CHUNK * 8) / 1024);
    return true;
  } else {
    Serial.println("[ERROR] No PSRAM detected! This code requires PSRAM.");
    return false;
  }
}

// ====== SD Card Management ======
void readFilesIntoList() {
  fileList.clear();
  if (!sd_ok) { 
    fileList.push_back("NO SD!"); 
    selectedIndex = 0; 
    scrollOffset = 0; 
    needsDisplayUpdate = true; 
    return; 
  }
  
  SDGuard lock(sdMutex);
  File root = SD.open("/");
  while (true) {
    File entry = root.openNextFile();
    if (!entry) break;
    if (!entry.isDirectory()) {
      String n = entry.name(); 
      String nlow = n; 
      nlow.toLowerCase();
      if (nlow.endsWith(".wav")) fileList.push_back(n);
    }
    entry.close();
  }
  root.close();
  
  if (selectedIndex >= (int)fileList.size()) 
    selectedIndex = max(0, (int)fileList.size() - 1);
  if (scrollOffset > selectedIndex) 
    scrollOffset = selectedIndex;
  needsDisplayUpdate = true;
}

void updateFileIndexOnBoot() {
  uint32_t maxIdx = 0;
  for (auto &name : fileList) {
    int idx = name.indexOf("Sample-"); 
    int end = name.lastIndexOf(".wav");
    if (idx >= 0 && end > idx) { 
      int num = name.substring(idx+7, end).toInt(); 
      if (num > maxIdx) maxIdx = num; 
    }
  }
  fileIndex = maxIdx + 1;
}

String getNextFilename() {
  String path;
  while (true) { 
    path = "/Sample-" + String(fileIndex++) + ".wav"; 
    SDGuard lock(sdMutex); 
    if (!SD.exists(path)) break; 
  }
  return path;
}

void trySDInit() {
  Serial.println("[SD] Attempting SD init...");
  
  {
    SDGuard lock(sdMutex); 
    sd_ok = SD.begin(SD_CS, SPI, 25000000);
  }
  
  Serial.print(F("SD init: ")); 
  Serial.println(sd_ok ? F("OK") : F("FAILED"));
  
  if (sd_ok) { 
    readFilesIntoList();
    updateFileIndexOnBoot(); 
  } else { 
    fileList.clear(); 
    fileList.push_back("NO SD!"); 
    selectedIndex = 0; 
    scrollOffset = 0; 
  }
  needsDisplayUpdate = true;
}

// ====== WAV Header ======
void patchWavHeader(File &file, uint32_t totalBytes) {
  uint32_t sampleRate = SAMPLE_RATE; 
  uint16_t bitsPerSample = WAV_BITS_PER_SAMPLE; 
  uint16_t numChannels = 1;
  uint32_t byteRate = sampleRate * numChannels * bitsPerSample / 8;
  uint16_t blockAlign = numChannels * bitsPerSample / 8;
  uint32_t dataChunkSize = totalBytes; 
  uint32_t chunkSize = 36 + dataChunkSize;
  
  file.seek(0); 
  
  file.write((const uint8_t*)"RIFF", 4); 
  file.write((uint8_t *)&chunkSize, 4);
  file.write((const uint8_t*)"WAVE", 4); 
  
  file.write((const uint8_t*)"fmt ", 4);
  uint32_t subChunk1Size = 16; 
  uint16_t audioFormat = 1;
  file.write((uint8_t *)&subChunk1Size, 4); 
  file.write((uint8_t *)&audioFormat, 2);
  file.write((uint8_t *)&numChannels, 2); 
  file.write((uint8_t *)&sampleRate, 4);
  file.write((uint8_t *)&byteRate, 4); 
  file.write((uint8_t *)&blockAlign, 2);
  file.write((uint8_t *)&bitsPerSample, 2); 
  
  file.write((const uint8_t*)"data", 4); 
  file.write((uint8_t *)&dataChunkSize, 4);
  
  file.flush();
}

// ====== OLED UI ======
void updateDisplay() {
  display.clearDisplay();
  if (currentScreen == UIScreen::SAMPLER) { 
    drawSamplerScreen(); 
  } else { 
    drawSamplesBrowser(); 
  }
  display.display();
}

void drawSamplesBrowser() {
  display.setTextSize(2); 
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0); 
  display.println(F("Samples:"));
  display.setTextSize(1);
  
  int line = 20;
  for (int i = scrollOffset; i < (int)fileList.size() && (i - scrollOffset) < maxVisibleItems; i++) {
    display.setCursor(0, line);
    if (i == selectedIndex) 
      display.setTextColor(SSD1306_BLACK, SSD1306_WHITE);
    else                     
      display.setTextColor(SSD1306_WHITE);
    String name = fileList[i];
    if (name.length() > 13) 
      name = name.substring(0, 12) + "...";
    display.print(name);
    line += 10;
  }
  
  if (showSideMenu && sd_ok) {
    display.drawLine(78, 20, 78, 20 + (maxVisibleItems * 10), SSD1306_WHITE);
    static const char* menuItems[] = {"Play", "Assign", "Delete", "Rename", "Back"};
    constexpr int menuCount = sizeof(menuItems) / sizeof(menuItems[0]);
    for (int i = sideMenuScrollOffset; i < menuCount && (i - sideMenuScrollOffset) < maxVisibleItems; i++) {
      int y = 20 + (i - sideMenuScrollOffset) * 10;
      display.setCursor(82, y);
      if (i == sideMenuIndex) 
        display.setTextColor(SSD1306_BLACK, SSD1306_WHITE);
      else                    
        display.setTextColor(SSD1306_WHITE);
      display.print(menuItems[i]);
    }
  }
}

void drawSamplerScreen() {
  display.setTextSize(2); 
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(0, 0); 
  display.println(F("Sampler"));
  
  if (psram_available) {
    display.setTextSize(1);
    display.setCursor(85, 0);
    display.print(F("8MB PSRAM"));
    display.setCursor(85, 8);
    display.printf("%dV", NUM_VOICES);
  }
  
  display.setTextSize(1);
  const String name = assignedSampleName.length() ? assignedSampleName : String(F("No Sample"));
  if (assignedNameDirty) {
    int16_t x1, y1;
    display.getTextBounds(name, 0, 0, &x1, &y1, &assignedNameW, &assignedNameH);
    assignedNameDirty = false;
  }
  int boxPaddingX = 3, boxPaddingY = 2;
  int boxW = min((int)SCREEN_WIDTH - 4, (int)assignedNameW + 2*boxPaddingX);
  int boxH = assignedNameH + 2*boxPaddingY;
  int boxX = 2, boxY = SCREEN_HEIGHT - boxH - 2;
  if (samplerFocusIndex == 0) {
    display.drawRect(boxX-1, boxY-1, boxW+2, boxH+2, SSD1306_WHITE);
  }
  display.drawRect(boxX, boxY, boxW, boxH, SSD1306_WHITE);
  display.setCursor(boxX + boxPaddingX, boxY + boxPaddingY);
  display.setTextColor(SSD1306_WHITE);
  display.print(name);
}

// ====== Rotary Encoder ======
void pollRotaryAndButton() {
  if (millis() - lastEncoderPoll < 5) return;
  lastEncoderPoll = millis();
  
  bool clk = digitalRead(ROTARY_CLK);
  bool dt = digitalRead(ROTARY_DT);
  bool btn = digitalRead(ROTARY_SW);
  int listMax = (sd_ok ? fileList.size() : 1);
  
  if (clk != lastClk && clk == LOW) {
    if (dt != clk) {
      if (currentScreen == UIScreen::BROWSER) {
        if (showSideMenu && sd_ok) { 
          if (sideMenuIndex < maxSideMenuItems - 1) { 
            sideMenuIndex++; 
            if (sideMenuIndex >= sideMenuScrollOffset + maxVisibleItems) 
              sideMenuScrollOffset++; 
          }
        } else { 
          if (selectedIndex < listMax - 1) { 
            selectedIndex++; 
            if (selectedIndex >= scrollOffset + maxVisibleItems) 
              scrollOffset++; 
          }
        }
      }
    } else {
      if (currentScreen == UIScreen::BROWSER) {
        if (showSideMenu && sd_ok) { 
          if (sideMenuIndex > 0) { 
            sideMenuIndex--; 
            if (sideMenuIndex < sideMenuScrollOffset) 
              sideMenuScrollOffset--; 
          }
        } else { 
          if (selectedIndex > 0) { 
            selectedIndex--; 
            if (selectedIndex < scrollOffset) 
              scrollOffset--; 
          }
        }
      }
    }
    needsDisplayUpdate = true;
  }
  lastClk = clk;
  
  if (currentScreen == UIScreen::SAMPLER && btn == LOW && lastRotaryBtn == HIGH) {
    if (samplerFocusIndex == 0) {
      currentScreen = UIScreen::BROWSER; 
      showSideMenu = false; 
      sideMenuIndex = 0; 
      sideMenuScrollOffset = 0;
      menuState = MenuState::WaitRelease; 
      needsDisplayUpdate = true; 
      lastRotaryBtn = btn; 
      return;
    }
  }
  
  if (currentScreen == UIScreen::BROWSER) {
    if (!showSideMenu && menuState == MenuState::WaitRelease && btn == HIGH) 
      menuState = MenuState::Accept;
    
    if (!showSideMenu && menuState == MenuState::Ready && btn == LOW && lastRotaryBtn == HIGH) {
      if (sd_ok) { 
        showSideMenu = true; 
        sideMenuIndex = 0; 
        sideMenuScrollOffset = 0; 
        needsDisplayUpdate = true; 
        menuState = MenuState::WaitRelease; 
      }
    } else if (!showSideMenu && menuState == MenuState::Accept && btn == LOW && lastRotaryBtn == HIGH) {
      showSideMenu = true; 
      sideMenuIndex = 0; 
      sideMenuScrollOffset = 0; 
      needsDisplayUpdate = true; 
      menuState = MenuState::WaitRelease;
    }
    
    if (menuState == MenuState::WaitRelease && btn == HIGH && showSideMenu) 
      menuState = MenuState::Accept;
    
    if (showSideMenu && menuState == MenuState::Accept && btn == LOW && lastRotaryBtn == HIGH) {
      if (sd_ok) {
        switch (sideMenuIndex) {
          case 0: // Play
            if (state == AppState::IDLE && fileList.size() > 0) { 
              playbackFilename = fileList[selectedIndex]; 
              isPlaying = true; 
              state = AppState::PLAYING; 
              mixerEnabled = false; 
              xTaskCreatePinnedToCore(playbackTask, "Playback", 32768, NULL, 3, &playbackTaskHandle, 1); 
            } 
            showSideMenu = false; 
            break;
          case 1: // Assign
            if (fileList.size() > 0) { 
              assignedSampleName = fileList[selectedIndex]; 
              assignedNameDirty = true; 
              resetVoicesAndParser(); 
            } else { 
              assignedSampleName = "No Sample"; 
              assignedNameDirty = true; 
              resetVoicesAndParser(); 
            } 
            showSideMenu = false; 
            currentScreen = UIScreen::SAMPLER; 
            break;
          case 2: // Delete
            if (state == AppState::IDLE && fileList.size() > 0) { 
              String fn = "/" + fileList[selectedIndex]; 
              { 
                SDGuard lock(sdMutex); 
                SD.remove(fn); 
              } 
              readFilesIntoList(); 
              if (selectedIndex >= (int)fileList.size()) { 
                selectedIndex = max(0, (int)fileList.size() - 1); 
                if (scrollOffset > selectedIndex) 
                  scrollOffset = selectedIndex; 
              }
            } 
            showSideMenu = false; 
            break;
          case 3: // Rename
            showSideMenu = false; 
            break;
          case 4: // Back
            showSideMenu = false; 
            break;
        }
      }
      needsDisplayUpdate = true; 
      menuState = MenuState::Ready;
    }
  }
  lastRotaryBtn = btn;
}

// ====== I2S Management ======
void i2s_uninstall_safe() { 
  esp_err_t result = i2s_driver_uninstall(I2S_NUM_0);
  if (result == ESP_OK) {
    Serial.println("[I2S] Driver uninstalled successfully");
  }
  vTaskDelay(pdMS_TO_TICKS(50));
}

void i2s_apply_profile(bool tx, bool stereo) {
  i2s_config_t cfg = {
    .mode = (i2s_mode_t)(I2S_MODE_MASTER | (tx ? I2S_MODE_TX : I2S_MODE_RX)), 
    .sample_rate = SAMPLE_RATE,
    .bits_per_sample = tx ? PLAY_TX_BITS : RECORD_RX_BITS, 
    .channel_format = stereo ? I2S_CHANNEL_FMT_RIGHT_LEFT : I2S_CHANNEL_FMT_ONLY_LEFT,
    .communication_format = I2S_COMM_FORMAT_STAND_I2S, 
    .intr_alloc_flags = ESP_INTR_FLAG_LEVEL1,
    .dma_buf_count = tx ? 16 : 8,
    .dma_buf_len = 1024, // CORRECTED: Valid DMA buffer size
    .use_apll = true, 
    .tx_desc_auto_clear = true, 
    .fixed_mclk = 0
  };
  
  i2s_pin_config_t pins = { 
    .bck_io_num = I2S_SCK, 
    .ws_io_num = I2S_WS, 
    .data_out_num = tx ? I2S_DAC_DOUT : I2S_PIN_NO_CHANGE, 
    .data_in_num  = tx ? I2S_PIN_NO_CHANGE : I2S_MIC_SD 
  };
  
  i2s_driver_install(I2S_NUM_0, &cfg, 0, NULL);
  i2s_set_pin(I2S_NUM_0, &pins);
  i2s_zero_dma_buffer(I2S_NUM_0);
}

// ====== Task Synchronization ======
bool stopMixerTaskSafely() {
  if (mixerTaskHandle == NULL) {
    return true;
  }
  
  Serial.println("[SYNC] Requesting mixer stop...");
  requestMixerStop = true;
  
  uint32_t timeout_start = millis();
  while (!mixerTaskStopped && (millis() - timeout_start < 1000)) {
    vTaskDelay(pdMS_TO_TICKS(5));
  }
  
  if (mixerTaskStopped) {
    Serial.println("[SYNC] Mixer stopped gracefully");
    mixerTaskHandle = NULL;
    vTaskDelay(pdMS_TO_TICKS(50));
    return true;
  } else {
    if (mixerTaskHandle != NULL) {
      vTaskDelete(mixerTaskHandle);
      mixerTaskHandle = NULL;
    }
    mixerTaskStopped = true;
    vTaskDelay(pdMS_TO_TICKS(100));
    return false;
  }
}

void startMixerTaskSafely() {
  if (mixerTaskHandle != NULL) {
    return;
  }
  
  requestMixerStop = false;
  mixerTaskStopped = false;
  
  portENTER_CRITICAL(&voicesMux);
  for (int v = 0; v < NUM_VOICES; v++) {
    if (voices[v].state == VoiceState::Playing) {
      voices[v].state = VoiceState::Inactive;
      voices[v].file_valid = false;
    }
  }
  portEXIT_CRITICAL(&voicesMux);
  
  BaseType_t result = xTaskCreatePinnedToCore(mixerTask, "MixerTask", 65536, NULL, 5, &mixerTaskHandle, 1);
  if (result == pdPASS) {
    Serial.println("[SYNC] Mixer task started");
    vTaskDelay(pdMS_TO_TICKS(200));
  }
}

// ====== Dynamic Buffer Threshold ======
size_t calculateRefillThreshold(float phaseInc) {
  float ratio = phaseInc;
  if (ratio < 0.5f) ratio = 0.5f;
  
  size_t threshold;
  if (ratio <= 1.0f) {
    threshold = BASE_REFILL_THRESHOLD;
  } else if (ratio <= 2.0f) {
    threshold = (size_t)(BASE_REFILL_THRESHOLD * ratio * 2.0f);
  } else {
    threshold = (size_t)(BASE_REFILL_THRESHOLD * powf(ratio, 2.0f) * 1.5f);
  }
  
  if (threshold < MIN_REFILL_THRESHOLD) threshold = MIN_REFILL_THRESHOLD;
  if (threshold > MAX_REFILL_THRESHOLD) threshold = MAX_REFILL_THRESHOLD;
  
  return threshold;
}

// ====== Recording Functions ======
bool startRecording() {
  String filename = getNextFilename(); 
  Serial.printf("[REC] Creating file: %s\n", filename.c_str());
  
  { 
    SDGuard lock(sdMutex); 
    recordingFile = SD.open(filename, FILE_WRITE); 
  }
  
  if (!recordingFile) { 
    Serial.println(F("[REC] File open FAILED")); 
    return false; 
  }
  
  uint8_t headerBuffer[WAV_HEADER_SIZE];
  memset(headerBuffer, 0, WAV_HEADER_SIZE);
  recordingFile.write(headerBuffer, WAV_HEADER_SIZE);
  recordingFile.flush();
  totalBytesWritten = 0; 
  
  return true;
}

void finishRecording() {
  { 
    SDGuard lock(sdMutex); 
    
    if (!recordingFile) {
      return;
    }
    
    recordingFile.flush();
    patchWavHeader(recordingFile, totalBytesWritten); 
    recordingFile.flush();
    recordingFile.close(); 
  }
  
  readFilesIntoList();
}

void recordAudioLoop() {
  size_t bytesRead = 0;
  
  i2s_read(I2S_NUM_0, rxBuffer, RECORD_BUFFER_SIZE, &bytesRead, pdMS_TO_TICKS(100));
  
  if (bytesRead == 0) {
    return;
  }
  
  size_t outIndex = 0;
  for (size_t i = 0; i + 3 < bytesRead; i += 4) {
    int32_t s32 = ((int32_t)rxBuffer[i+3] << 24) | ((int32_t)rxBuffer[i+2] << 16) | 
                  ((int32_t)rxBuffer[i+1] << 8) | rxBuffer[i+0];
    
    int16_t s16 = (int16_t)(s32 >> 16);
    
    wavOutBuffer[outIndex++] = (uint8_t)(s16 & 0xFF); 
    wavOutBuffer[outIndex++] = (uint8_t)((s16 >> 8) & 0xFF);
  }
  
  if (outIndex > 0) { 
    SDGuard lock(sdMutex); 
    size_t written = recordingFile.write(wavOutBuffer, outIndex); 
    if (written > 0) {
      totalBytesWritten += written; 
    }
    
    static uint32_t flushCounter = 0;
    if (++flushCounter >= 10) {
      recordingFile.flush();
      flushCounter = 0;
    }
  }
}

void handleRecordingButton() {
  int buttonState = digitalRead(BUTTON_PIN); 
  unsigned long now = millis();
  
  if (sd_ok && buttonState == LOW && !wasRecordPressed && state == AppState::IDLE && 
      (now - lastRecordDebounce > recordDebounceDelay)) {
    
    delay(recordDebounceDelay);
    if (digitalRead(BUTTON_PIN) == LOW) {
      lastRecordDebounce = now; 
      wasRecordPressed = true; 
      digitalWrite(LED_PIN, HIGH);
      
      mixerEnabled = false;
      stopMixerTaskSafely();
      i2s_uninstall_safe();
      i2s_apply_profile(false, false);
      
      if (!startRecording()) {
        digitalWrite(LED_PIN, LOW);
        i2s_uninstall_safe();
        i2s_apply_profile(true, true);
        startMixerTaskSafely();
        mixerEnabled = true;
        return;
      }
      
      state = AppState::RECORDING; 
      stateStart = millis();
    }
  }
  
  if (buttonState == HIGH) wasRecordPressed = false;
}

// ====== Preview Playback ======
void playbackTask(void *param) {
  String fn = playbackFilename; 
  if (fn.length() > 0 && fn[0] != '/') fn = "/" + fn;
  
  File file; 
  { 
    SDGuard lock(sdMutex); 
    file = SD.open(fn); 
  }
  
  if (!file || file.size() <= WAV_HEADER_SIZE) {
    if (file) file.close(); 
    isPlaying = false; 
    state = AppState::IDLE; 
    mixerEnabled = true; 
    vTaskDelete(NULL); 
    return;
  }
  
  file.seek(WAV_HEADER_SIZE);
  
  while (isPlaying) {
    size_t nRead; 
    { 
      SDGuard lock(sdMutex); 
      nRead = file.read(preview_rb, PREVIEW_READ_BUF); 
    }
    
    if (nRead == 0) { 
      break; 
    }
    
    size_t pos = 0;
    while (pos < nRead && isPlaying) {
      int frames = 0;
      while (frames < PREVIEW_FRAMES_PER_CHUNK && pos + 1 < nRead) {
        int16_t s = (int16_t)((preview_rb[pos+1] << 8) | preview_rb[pos]); 
        pos += 2;
        int32_t amp = (int32_t)((float)s * PREVIEW_PLAYBACK_GAIN);
        if (amp > 32767) amp = 32767; 
        if (amp < -32768) amp = -32768;
        int32_t s32 = ((int32_t)(int16_t)amp) << 16;
        preview_stereo32[2*frames] = s32; 
        preview_stereo32[2*frames + 1] = s32; 
        frames++;
      }
      if (frames > 0) { 
        size_t bytesWritten; 
        i2s_write(I2S_NUM_0, (const char*)preview_stereo32, 
                  frames * 2 * sizeof(int32_t), &bytesWritten, portMAX_DELAY); 
      }
    }
  }
  
  file.close();
  isPlaying = false; 
  state = AppState::IDLE; 
  mixerEnabled = true; 
  needsDisplayUpdate = true; 
  vTaskDelete(NULL);
}

// ====== MIDI Handling ======
HardwareSerial &MIDI_SERIAL = Serial1;
volatile uint8_t  gRunStatus = 0, gNeeded = 0, gData1 = 0;
volatile uint32_t gLastByteMs = 0;

static inline uint8_t statusDataLen(uint8_t st) {
  uint8_t hi = st & 0xF0;
  if (hi == 0xC0 || hi == 0xD0) return 1;
  if (hi >= 0x80 && hi <= 0xE0) return 2;
  return 0;
}

int findFreeVoice() {
  for (int v = 0; v < NUM_VOICES; v++) 
    if (voices[v].state == VoiceState::Inactive) return v;
  
  uint32_t oldest = 0xFFFFFFFF; 
  int idx = 0;
  for (int v = 0; v < NUM_VOICES; v++) {
    if (voices[v].state == VoiceState::Playing && voices[v].startedAtMs < oldest) {
      oldest = voices[v].startedAtMs;
      idx = v;
    }
  }
  return idx;
}

void voiceNoteOn(uint8_t note, uint8_t vel) { 
  MidiMessage msg = { true, note, vel }; 
  xQueueSend(midiQueue, &msg, 0); 
}

void voiceNoteOff(uint8_t note) { 
  MidiMessage msg = { false, note, 0 }; 
  xQueueSend(midiQueue, &msg, 0); 
}

void handleMIDI() {
  uint32_t now = millis();
  if (now - gLastByteMs > 300) { 
    gRunStatus = 0; 
    gNeeded = 0; 
  }
  
  while (MIDI_SERIAL.available()) {
    uint8_t b = MIDI_SERIAL.read(); 
    gLastByteMs = now;
    
    if (b >= 0xF8) continue;
    
    if (b & 0x80) {
      if (b >= 0xF0) {
        gRunStatus = 0; 
        gNeeded = 0;
      } else {
        gRunStatus = b; 
        gNeeded = statusDataLen(b);
      }
      continue;
    }
    
    if (gRunStatus == 0 || gNeeded == 0) continue;
    
    if (gNeeded == 2) { 
      gData1 = b; 
      gNeeded = 1; 
    } else {
      uint8_t data2 = b;
      uint8_t st = gRunStatus & 0xF0;
      
      if (st == 0x90) {
        if (data2 > 0) voiceNoteOn(gData1, data2); 
        else voiceNoteOff(gData1); 
      } else if (st == 0x80) {
        voiceNoteOff(gData1); 
      }
      
      gNeeded = statusDataLen(gRunStatus);
    }
  }
}

void resetVoicesAndParser() {
  File filesToClose[NUM_VOICES];
  int closeCount = 0;
  
  portENTER_CRITICAL(&voicesMux);
  for (int v = 0; v < NUM_VOICES; v++) {
    if (voices[v].state != VoiceState::Inactive && voices[v].file_valid && voices[v].file) {
      filesToClose[closeCount++] = voices[v].file;
      voices[v].file = File();
    }
    voices[v].state = VoiceState::Inactive;
    voices[v].file_valid = false;
    voices[v].refill_threshold = BASE_REFILL_THRESHOLD;
  }
  portEXIT_CRITICAL(&voicesMux);
  
  if (closeCount > 0) {
    SDGuard lock(sdMutex);
    for (int i = 0; i < closeCount; i++) { 
      filesToClose[i].close(); 
    }
  }
  
  gRunStatus = 0; 
  gNeeded = 0; 
  gData1 = 0; 
  gLastByteMs = millis();
  while (MIDI_SERIAL.available()) (void)MIDI_SERIAL.read();
  if (midiQueue) xQueueReset(midiQueue);
}

void processMidiQueue() {
  MidiMessage msg;
  while (xQueueReceive(midiQueue, &msg, 0) == pdPASS) {
    if (msg.isNoteOn) { 
      triggerVoiceOn(msg.note, msg.velocity); 
    } else { 
      triggerVoiceOff(msg.note); 
    }
  }
}

void triggerVoiceOn(uint8_t note, uint8_t vel) {
  if (assignedSampleName == "No Sample" || assignedSampleName.length() == 0) return;
  
  float ratio = powf(2.0f, (note - MIDI_ROOT_NOTE) / 12.0f);
  
  String path = "/" + assignedSampleName;
  File newFile;
  bool fileOpened = false;
  
  {
    SDGuard lock(sdMutex);
    newFile = SD.open(path, FILE_READ);
    if (newFile) {
      newFile.seek(WAV_HEADER_SIZE);
      fileOpened = true;
    }
  }
  
  if (!fileOpened) {
    return;
  }
  
  File fileToClose;
  bool shouldClose = false;
  int v_idx;
  
  portENTER_CRITICAL(&voicesMux);
  v_idx = findFreeVoice();
  if (voices[v_idx].state == VoiceState::Playing && voices[v_idx].file_valid) {
    fileToClose = voices[v_idx].file;
    shouldClose = true;
  }
  
  voices[v_idx].file = newFile;
  voices[v_idx].file_valid = true;
  voices[v_idx].state = VoiceState::Playing;
  voices[v_idx].note = note;
  voices[v_idx].phase = 0.0f;
  voices[v_idx].phaseInc = ratio;
  voices[v_idx].startedAtMs = millis();
  voices[v_idx].buffer_len = 0;
  voices[v_idx].needs_refill = true;
  voices[v_idx].is_eof = false;
  voices[v_idx].refill_threshold = calculateRefillThreshold(ratio);
  voices[v_idx].last_phase_inc = ratio;
  portEXIT_CRITICAL(&voicesMux);
  
  if (shouldClose) {
    SDGuard lock(sdMutex);
    fileToClose.close();
  }
}

void triggerVoiceOff(uint8_t note) {
  portENTER_CRITICAL(&voicesMux);
  for (int v = 0; v < NUM_VOICES; v++) {
    if (voices[v].state == VoiceState::Playing && voices[v].note == note) {
      voices[v].state = VoiceState::Finished;
    }
  }
  portEXIT_CRITICAL(&voicesMux);
}

void refillVoiceBuffers() {
  static uint32_t lastRefillCheck = 0;
  if (millis() - lastRefillCheck < 2) return;
  lastRefillCheck = millis();
  
  bool needs_sd_access = false;
  
  portENTER_CRITICAL(&voicesMux);
  for(int v = 0; v < NUM_VOICES; v++) { 
    if(voices[v].state == VoiceState::Playing && voices[v].needs_refill && 
       !voices[v].is_eof && voices[v].file_valid) { 
      needs_sd_access = true; 
      break;
    } 
  }
  portEXIT_CRITICAL(&voicesMux);
  
  if (!needs_sd_access) return;
  
  SDGuard lock(sdMutex);
  
  for (int v = 0; v < NUM_VOICES; v++) {
    portENTER_CRITICAL(&voicesMux);
    bool shouldRefill = (voices[v].state == VoiceState::Playing && 
                        voices[v].needs_refill && 
                        !voices[v].is_eof && 
                        voices[v].file_valid);
    
    if (shouldRefill && voices[v].last_phase_inc != voices[v].phaseInc) {
      voices[v].refill_threshold = calculateRefillThreshold(voices[v].phaseInc);
      voices[v].last_phase_inc = voices[v].phaseInc;
    }
    portEXIT_CRITICAL(&voicesMux);
    
    if (shouldRefill) {
      size_t bytesRead = voices[v].file.read((uint8_t*)voices[v].buffer, VOICE_BUFFER_BYTES);
      
      portENTER_CRITICAL(&voicesMux);
      if (bytesRead > 0) { 
        voices[v].buffer_len = bytesRead / sizeof(int16_t); 
        voices[v].phase = 0;
        
        if (bytesRead < VOICE_BUFFER_BYTES) {
          voices[v].is_eof = true;
          for(size_t i = bytesRead / sizeof(int16_t); i < VOICE_BUFFER_SAMPLES; i++) { 
            voices[v].buffer[i] = 0; 
          }
        }
      } else {
        voices[v].is_eof = true;
        voices[v].buffer_len = 0;
      }
      voices[v].needs_refill = false;
      portEXIT_CRITICAL(&voicesMux);
    }
  }
}

void cleanupFinishedVoices() {
  File filesToClose[NUM_VOICES];
  int closeCount = 0;
  
  portENTER_CRITICAL(&voicesMux);
  for (int v = 0; v < NUM_VOICES; v++) {
    if (voices[v].state == VoiceState::Finished) {
      if (voices[v].file_valid && voices[v].file) {
        filesToClose[closeCount++] = voices[v].file;
      }
      voices[v].state = VoiceState::Inactive;
      voices[v].file = File();
      voices[v].file_valid = false;
      voices[v].refill_threshold = BASE_REFILL_THRESHOLD;
    }
  }
  portEXIT_CRITICAL(&voicesMux);
  
  if (closeCount > 0) {
    SDGuard lock(sdMutex);
    for (int i = 0; i < closeCount; i++) {
      filesToClose[i].close();
    }
  }
}

void mixerTask(void *param) {
  i2s_uninstall_safe();
  vTaskDelay(pdMS_TO_TICKS(100));
  i2s_apply_profile(true, true);
  
  const int outStereoCount = MIXER_FRAMES_PER_CHUNK * 2;
  int32_t *stereo32 = (int32_t*)heap_caps_malloc(outStereoCount * sizeof(int32_t), MALLOC_CAP_SPIRAM);
  
  if (!stereo32) { 
    mixerTaskStopped = true;
    vTaskDelete(NULL); 
    return; 
  }
  
  while (true) {
    if (requestMixerStop) {
      heap_caps_free(stereo32);
      mixerTaskStopped = true;
      vTaskDelete(NULL);
      return;
    }
    
    if (!mixerEnabled) { 
      vTaskDelay(pdMS_TO_TICKS(5)); 
      continue; 
    }
    
    memset(stereo32, 0, outStereoCount * sizeof(int32_t));
    
    for (int n = 0; n < MIXER_FRAMES_PER_CHUNK; n++) {
      int32_t mix = 0;
      int active_voices = 0;
      
      portENTER_CRITICAL(&voicesMux);
      for (int v = 0; v < NUM_VOICES; v++) {
        if (voices[v].state != VoiceState::Playing || !voices[v].file_valid) continue;
        
        active_voices++;
        
        float remaining_samples = voices[v].buffer_len - voices[v].phase;
        if (!voices[v].needs_refill && remaining_samples <= voices[v].refill_threshold) { 
          voices[v].needs_refill = true; 
        }
        
        float p = voices[v].phase; 
        int idx = (int)p; 
        
        if (idx >= (int)voices[v].buffer_len - 1) {
          if (voices[v].is_eof) {
            voices[v].state = VoiceState::Finished;
          }
          continue;
        }
        
        int16_t s0 = voices[v].buffer[idx]; 
        int16_t s1 = (idx + 1 < (int)voices[v].buffer_len) ? voices[v].buffer[idx + 1] : s0;
        
        float frac = p - idx;
        float sample = (1.0f - frac) * (float)s0 + frac * (float)s1;
        
        mix += (int32_t)(sample * 0.8f);
        
        voices[v].phase += voices[v].phaseInc;
      }
      portEXIT_CRITICAL(&voicesMux);
      
      if (active_voices > 0) {
        mix = mix / max(1, (active_voices / 4));
      }
      
      if (mix > 28000) mix = 28000; 
      if (mix < -28000) mix = -28000;
      
      int32_t s32 = ((int32_t)(int16_t)mix) << 16;
      stereo32[2*n + 0] = s32; 
      stereo32[2*n + 1] = s32;
    }
    
    size_t bytesWritten;
    i2s_write(I2S_NUM_0, (const char*)stereo32, 
              outStereoCount * sizeof(int32_t), &bytesWritten, pdMS_TO_TICKS(20));
    
    taskYIELD();
  }
}

void drawTextCentered(const String &txt, int16_t cx, int16_t cy, uint8_t size) {
  int16_t x1, y1; 
  uint16_t w, h; 
  display.setTextSize(size); 
  display.setTextColor(SSD1306_WHITE);
  display.getTextBounds(txt, 0, 0, &x1, &y1, &w, &h);
  display.setCursor(cx - (int16_t)w / 2, cy - (int16_t)h / 2); 
  display.print(txt);
}

void showStartupMenu() {
  display.clearDisplay();
  drawTextCentered(F("Kas-8 Sampler"), SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 - 10, 1);
  drawTextCentered(F("ESP32-S3 8MB"), SCREEN_WIDTH / 2, SCREEN_HEIGHT / 2 + 5, 1);
  
  display.setTextSize(1); 
  display.setTextColor(SSD1306_WHITE);
  display.setCursor(2, SCREEN_HEIGHT - 10); 
  display.print(F("V4.0 PSRAM"));
  
  display.display(); 
  delay(2000); 
  display.clearDisplay(); 
  display.display();
}

// ====== MAIN SETUP ======
void setup() {
  Serial.begin(115200);
  delay(100);
  Serial.println("\n=== Kas-8 Sampler V4.0 - ESP32-S3 with 8MB PSRAM ===");
  
  // Allocate PSRAM buffers first
  if (!allocatePSRAMBuffers()) {
    Serial.println("[ERROR] Failed to allocate PSRAM buffers!");
    while(1) { delay(1000); }
  }
  
  // Pin setup
  pinMode(BUTTON_PIN, INPUT_PULLUP); 
  pinMode(LED_PIN, OUTPUT); 
  pinMode(ROTARY_CLK, INPUT);
  pinMode(ROTARY_DT, INPUT); 
  pinMode(ROTARY_SW, INPUT_PULLUP); 
  digitalWrite(LED_PIN, LOW);

  // MIDI setup
  midiQueue = xQueueCreate(256, sizeof(MidiMessage));
  MIDI_SERIAL.setRxBufferSize(4096);
  pinMode(MIDI_RX_PIN, INPUT_PULLUP);
  MIDI_SERIAL.begin(31250, SERIAL_8N1, MIDI_RX_PIN, -1);
  
  // Semaphores
  mixerStopSemaphore = xSemaphoreCreateBinary();
  sdMutex = xSemaphoreCreateMutex();
  
  // Display setup
  I2C_BUS.begin(OLED_SDA, OLED_SCL, 400000);
  if (!display.begin(SSD1306_SWITCHCAPVCC, 0x3C)) { 
    Serial.println(F("OLED failed"));
  }
  showStartupMenu();
  
  // SD card setup
  SPI.begin(12, 13, 11);  // SCK, MISO, MOSI
  trySDInit();
  
  // Initialize note ratios
  for (int n = 0; n < 128; n++) { 
    gNoteRatio[n] = powf(2.0f, (n - MIDI_ROOT_NOTE) / 12.0f); 
  }
  
  // Initialize voices
  for (int v = 0; v < NUM_VOICES; v++) {
    voices[v].refill_threshold = BASE_REFILL_THRESHOLD;
  }
  
  // Start mixer
  startMixerTaskSafely();
  delay(300);
  
  resetVoicesAndParser();
  
  Serial.println(F("Setup complete"));
  Serial.printf("Free heap: %d bytes\n", ESP.getFreeHeap());
  Serial.printf("Free PSRAM: %d bytes\n", ESP.getFreePsram());
  
  needsDisplayUpdate = true;
}

// ====== MAIN LOOP ======
void loop() {
  handleMIDI();
  processMidiQueue();
  refillVoiceBuffers();
  cleanupFinishedVoices();

  if (!sd_ok && (millis() - lastSDRetry > 3000)) { 
    lastSDRetry = millis(); 
    trySDInit(); 
  }
  
  pollRotaryAndButton();
  
  if (needsDisplayUpdate) { 
    needsDisplayUpdate = false; 
    updateDisplay(); 
  }

  // State machine
  switch (state) {
    case AppState::IDLE:
      handleRecordingButton();
      break;
      
    case AppState::RECORDING: {
      bool stillHeld = (digitalRead(BUTTON_PIN) == LOW);
      if (stillHeld) {
        recordAudioLoop();
      } else {
        digitalWrite(LED_PIN, LOW);
        finishRecording();
        i2s_uninstall_safe();
        i2s_apply_profile(true, true);
        startMixerTaskSafely();
        mixerEnabled = true;
        
        if (mixerTaskHandle != NULL) {
          state = AppState::COOLING; 
          stateStart = millis();
        } else {
          state = AppState::IDLE;
        }
      }
      break;
    }
    
    case AppState::COOLING:
      if (millis() - stateStart >= 1500) {
        state = AppState::BLINKING; 
        blinkStart = millis(); 
        blinkState = 0; 
        blinkCount = 0;
      }
      break;
      
    case AppState::BLINKING:
      if (millis() - blinkStart >= 100) {
        blinkStart = millis(); 
        digitalWrite(LED_PIN, blinkState ? LOW : HIGH);
        blinkState = !blinkState; 
        if (!blinkState) {
          blinkCount++;
          if (blinkCount >= 3) {
            digitalWrite(LED_PIN, LOW);
            state = AppState::IDLE;
          }
        }
      }
      break;
      
    case AppState::PLAYING:
      break;
  }
  
  vTaskDelay(pdMS_TO_TICKS(1));
}
