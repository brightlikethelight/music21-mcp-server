# Security Policy

## Supported Versions

We release patches for security vulnerabilities. Which versions are eligible for receiving such patches depends on the CVSS v3.0 Rating:

| Version | Supported          |
| ------- | ------------------ |
| 1.0.x   | :white_check_mark: |
| < 1.0   | :x:                |

## Reporting a Vulnerability

We take the security of music21-mcp-server seriously. If you believe you have found a security vulnerability, please report it to us as described below.

**Please do not report security vulnerabilities through public GitHub issues.**

Instead, please report them via email to: brightliu@college.harvard.edu

You should receive a response within 48 hours. If for some reason you do not, please follow up via email to ensure we received your original message.

Please include the following information:
- Type of issue (e.g., buffer overflow, SQL injection, cross-site scripting, etc.)
- Full paths of source file(s) related to the manifestation of the issue
- The location of the affected source code (tag/branch/commit or direct URL)
- Any special configuration required to reproduce the issue
- Step-by-step instructions to reproduce the issue
- Proof-of-concept or exploit code (if possible)
- Impact of the issue, including how an attacker might exploit the issue

## Security Considerations

### MCP Protocol Security
- The MCP (Model Context Protocol) interface should only be exposed to trusted clients
- Authentication and authorization must be implemented at the deployment layer
- Never expose MCP endpoints directly to the internet

### HTTP API Security
- Always use HTTPS in production
- Implement rate limiting to prevent abuse
- Validate all input parameters
- Sanitize file paths to prevent directory traversal

### File System Security
- The server can read and write music files
- Ensure proper file permissions on the host system
- Consider running in a containerized environment with limited file access
- Never run with elevated privileges unless absolutely necessary

### Dependency Security
- We use automated dependency scanning via pip-audit and safety
- Dependencies are regularly updated to patch known vulnerabilities
- All dependencies are pinned to specific versions for reproducibility

## Security Best Practices

1. **Principle of Least Privilege**: Run the server with minimal required permissions
2. **Input Validation**: All user inputs are validated before processing
3. **Error Handling**: Errors do not expose sensitive system information
4. **Logging**: Security events are logged for audit purposes
5. **Updates**: Keep the server and all dependencies up to date

## Known Security Limitations

1. **music21 Library**: Some music21 operations may execute system commands (e.g., for Lilypond). Ensure these external programs are from trusted sources.
2. **File Operations**: The server can read/write files. Proper OS-level permissions are crucial.
3. **Memory Usage**: Large music files can consume significant memory. Consider resource limits.

## Security Tools in CI/CD

Our CI/CD pipeline includes:
- **Bandit**: Static security analysis for Python
- **pip-audit**: Vulnerability scanning for dependencies
- **Safety**: Additional vulnerability database checks

All security scans must pass for code to be merged.