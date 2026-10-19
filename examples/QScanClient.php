<?php
/**
 * QScanClient.php
 * Robust, production-ready PHP integration client for the PDF QA Agent API.
 * Supports streaming upload, status polling, and result fetching.
 */

class QScanClient {
    private string $baseUrl;
    private ?string $apiKey;

    public function __construct(string $baseUrl = "http://localhost:8000", ?string $apiKey = null) {
        $this->baseUrl = rtrim($baseUrl, "/");
        $this->apiKey = $apiKey;
    }

    public function setApiKey(string $apiKey): void {
        $this->apiKey = $apiKey;
    }

    public function getApiKey(): ?string {
        return $this->apiKey;
    }

    /**
     * Uploads a PDF to the scanner backend.
     *
     * @param string $pdfPath Absolute or relative path to the PDF file
     * @return array Contains job_id, status ('pending'), and filename
     */
    public function uploadPdf(string $pdfPath): array {
        $this->ensureApiKey();

        if (!file_exists($pdfPath)) {
            throw new InvalidArgumentException("PDF file not found: {$pdfPath}");
        }

        $url = "{$this->baseUrl}/v1/pdfs/upload";
        $cFile = new CURLFile($pdfPath, 'application/pdf', basename($pdfPath));

        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_POST           => true,
            CURLOPT_POSTFIELDS     => ['file' => $cFile],
            CURLOPT_HTTPHEADER     => [
                "X-API-Key: {$this->apiKey}",
                "Accept: application/json"
            ],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => 60
        ]);

        $response = curl_exec($ch);
        if ($response === false) {
            $err = curl_error($ch);
            curl_close($ch);
            throw new Exception("Upload failed: {$err}");
        }

        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        $data = json_decode($response, true);
        if ($httpCode !== 200) {
            $detail = is_array($data) ? json_encode($data["detail"] ?? $data) : $response;
            throw new Exception("Upload rejected (HTTP {$httpCode}): {$detail}");
        }

        return $data;
    }

    /**
     * Checks current processing status of a job.
     *
     * @param string $jobId
     * @return array Status payload with job_id, status, error
     */
    public function getStatus(string $jobId): array {
        $this->ensureApiKey();

        $url = "{$this->baseUrl}/v1/pdfs/" . urlencode($jobId) . "/status";
        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_HTTPHEADER     => [
                "X-API-Key: {$this->apiKey}",
                "Accept: application/json"
            ],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => 10
        ]);

        $response = curl_exec($ch);
        if ($response === false) {
            $err = curl_error($ch);
            curl_close($ch);
            throw new Exception("Status check failed: {$err}");
        }

        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        $data = json_decode($response, true);
        if ($httpCode === 404) {
            throw new Exception("Job not found or access denied for this client key.");
        }
        if ($httpCode !== 200) {
            $detail = is_array($data) ? json_encode($data["detail"] ?? $data) : $response;
            throw new Exception("Status check error (HTTP {$httpCode}): {$detail}");
        }

        return $data;
    }

    /**
     * Retrieves the completed extraction and validation JSON.
     *
     * IMPORTANT FIX: Reads HTTP status code strictly AFTER curl_exec.
     * - Returns ['status' => 'done', 'result' => [...]] on HTTP 200.
     * - Returns ['status' => ..., 'error' => ...] on HTTP 409 (still processing).
     *
     * @param string $jobId
     * @return array
     */
    public function getResult(string $jobId): array {
        $this->ensureApiKey();

        $url = "{$this->baseUrl}/v1/pdfs/" . urlencode($jobId) . "/result";
        $ch = curl_init($url);
        curl_setopt_array($ch, [
            CURLOPT_HTTPHEADER     => [
                "X-API-Key: {$this->apiKey}",
                "Accept: application/json"
            ],
            CURLOPT_RETURNTRANSFER => true,
            CURLOPT_TIMEOUT        => 15
        ]);

        // 1. Execute request FIRST
        $response = curl_exec($ch);

        // 2. Check transport/network error
        if ($response === false) {
            $err = curl_error($ch);
            curl_close($ch);
            throw new Exception("Result request failed: {$err}");
        }

        // 3. Read HTTP status code AFTER curl_exec
        $httpCode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
        curl_close($ch);

        $data = json_decode($response, true);

        // 4. Handle HTTP responses
        if ($httpCode === 200) {
            return [
                "status" => "done",
                "result" => $data
            ];
        }

        if ($httpCode === 409) {
            // Job still in progress (pending, extracting, correcting)
            $detail = $data["detail"] ?? [];
            return [
                "status" => $detail["status"] ?? "processing",
                "error"  => $detail["error"] ?? null
            ];
        }

        if ($httpCode === 404) {
            throw new Exception("Job not found or does not belong to this client key (HTTP 404).");
        }

        if ($httpCode === 401) {
            throw new Exception("Unauthorized: Invalid API key (HTTP 401).");
        }

        $detail = is_array($data) ? json_encode($data["detail"] ?? $data) : $response;
        throw new Exception("Unexpected HTTP {$httpCode}: {$detail}");
    }

    /**
     * Polls until the job completes or fails, with configurable timeout.
     *
     * @param string $jobId
     * @param int $pollIntervalSeconds Interval between poll checks (default: 5s)
     * @param int $timeoutSeconds Maximum wait time before giving up (default: 600s / 10m)
     * @return array The final result JSON on success
     */
    public function waitForResult(string $jobId, int $pollIntervalSeconds = 5, int $timeoutSeconds = 600): array {
        $startTime = time();

        while (true) {
            $res = $this->getResult($jobId);

            if ($res["status"] === "done") {
                return $res["result"];
            }

            if ($res["status"] === "failed") {
                $err = $res["error"] ?? "Unknown pipeline error";
                throw new Exception("PDF processing failed: {$err}");
            }

            if ((time() - $startTime) > $timeoutSeconds) {
                throw new Exception("Timed out waiting for job {$jobId} after {$timeoutSeconds} seconds.");
            }

            sleep($pollIntervalSeconds);
        }
    }

    private function ensureApiKey(): void {
        if (empty($this->apiKey)) {
            throw new RuntimeException("API key is not configured. Please supply an API key in the constructor or call setApiKey().");
        }
    }
}
