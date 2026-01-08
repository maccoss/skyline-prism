# Skyline-PRISM v26.1.2 Release Notes

## Overview

Maintenance release with line ending normalization and cleanup of deprecated configuration files.

## Changes

- **Line ending normalization**: Standardized all text files to use LF line endings (Unix style) instead of mixed CRLF/LF, improving cross-platform compatibility.
- **Cleanup**: Removed deprecated `prism-config.yaml` file. Use `config_template.yaml` instead, or generate a fresh template with `prism config-template`.

## Testing

- **291 tests passing**
- **60% overall coverage**

## Notes

This is a maintenance release with no functional changes to the PRISM pipeline. All algorithm behavior and API remain unchanged from v26.1.1.
