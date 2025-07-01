{{/*
Expand the name of the chart.
*/}}
{{- define "music21-mcp.name" -}}
{{- default .Chart.Name .Values.nameOverride | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Create a default fully qualified app name.
We truncate at 63 chars because some Kubernetes name fields are limited to this (by the DNS naming spec).
If release name contains chart name it will be used as a full name.
*/}}
{{- define "music21-mcp.fullname" -}}
{{- if .Values.fullnameOverride }}
{{- .Values.fullnameOverride | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- $name := default .Chart.Name .Values.nameOverride }}
{{- if contains $name .Release.Name }}
{{- .Release.Name | trunc 63 | trimSuffix "-" }}
{{- else }}
{{- printf "%s-%s" .Release.Name $name | trunc 63 | trimSuffix "-" }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Create chart name and version as used by the chart label.
*/}}
{{- define "music21-mcp.chart" -}}
{{- printf "%s-%s" .Chart.Name .Chart.Version | replace "+" "_" | trunc 63 | trimSuffix "-" }}
{{- end }}

{{/*
Common labels
*/}}
{{- define "music21-mcp.labels" -}}
helm.sh/chart: {{ include "music21-mcp.chart" . }}
{{ include "music21-mcp.selectorLabels" . }}
{{- if .Chart.AppVersion }}
app.kubernetes.io/version: {{ .Chart.AppVersion | quote }}
{{- end }}
app.kubernetes.io/managed-by: {{ .Release.Service }}
{{- end }}

{{/*
Selector labels
*/}}
{{- define "music21-mcp.selectorLabels" -}}
app.kubernetes.io/name: {{ include "music21-mcp.name" . }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{/*
Create the name of the service account to use
*/}}
{{- define "music21-mcp.serviceAccountName" -}}
{{- if .Values.serviceAccount.create }}
{{- default (include "music21-mcp.fullname" .) .Values.serviceAccount.name }}
{{- else }}
{{- default "default" .Values.serviceAccount.name }}
{{- end }}
{{- end }}

{{/*
Create the name of the secret to use
*/}}
{{- define "music21-mcp.secretName" -}}
{{- printf "%s-secrets" (include "music21-mcp.fullname" .) }}
{{- end }}

{{/*
Create the name of the configmap to use
*/}}
{{- define "music21-mcp.configMapName" -}}
{{- printf "%s-config" (include "music21-mcp.fullname" .) }}
{{- end }}

{{/*
Redis fullname
*/}}
{{- define "music21-mcp.redis.fullname" -}}
{{- include "redis.fullname" .Subcharts.redis }}
{{- end }}

{{/*
Redis secret name
*/}}
{{- define "music21-mcp.redis.secretName" -}}
{{- include "redis.secretName" .Subcharts.redis }}
{{- end }}

{{/*
Redis secret password key
*/}}
{{- define "music21-mcp.redis.secretPasswordKey" -}}
{{- include "redis.secretPasswordKey" .Subcharts.redis }}
{{- end }}

{{/*
Get the password secret.
*/}}
{{- define "music21-mcp.secretPassword" -}}
{{- if .Values.secrets.jwtSecret }}
    {{- .Values.secrets.jwtSecret }}
{{- else }}
    {{- randAlphaNum 32 }}
{{- end }}
{{- end }}

{{/*
Return the proper image name
*/}}
{{- define "music21-mcp.image" -}}
{{- $registryName := .Values.image.registry -}}
{{- $repositoryName := .Values.image.repository -}}
{{- $tag := .Values.image.tag | default .Chart.AppVersion -}}
{{- if .Values.global.imageRegistry }}
    {{- $registryName = .Values.global.imageRegistry -}}
{{- end -}}
{{- if $registryName }}
{{- printf "%s/%s:%s" $registryName $repositoryName $tag -}}
{{- else -}}
{{- printf "%s:%s" $repositoryName $tag -}}
{{- end -}}
{{- end }}

{{/*
Return the proper Docker Image Registry Secret Names
*/}}
{{- define "music21-mcp.imagePullSecrets" -}}
{{- if .Values.global.imagePullSecrets }}
{{- range .Values.global.imagePullSecrets }}
- name: {{ . }}
{{- end }}
{{- else if .Values.image.pullSecrets }}
{{- range .Values.image.pullSecrets }}
- name: {{ . }}
{{- end }}
{{- end }}
{{- end }}

{{/*
Return the proper Storage Class
*/}}
{{- define "music21-mcp.storageClass" -}}
{{- if .Values.global.storageClass -}}
{{- .Values.global.storageClass -}}
{{- else if .Values.persistence.storageClass -}}
{{- .Values.persistence.storageClass -}}
{{- end -}}
{{- end }}

{{/*
Validate configuration
*/}}
{{- define "music21-mcp.validateValues" -}}
{{- if and .Values.ingress.enabled (not .Values.ingress.hosts) -}}
music21-mcp: ingress.hosts
    You must provide at least one host when enabling ingress
{{- end -}}
{{- if and .Values.persistence.enabled (not .Values.persistence.size) -}}
music21-mcp: persistence.size
    You must provide a size when enabling persistence
{{- end -}}
{{- end }}