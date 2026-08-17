import { useState } from "react";
import { Check, Copy } from "./Icons.jsx";

function CodeBlock({ language, code }) {
  const [copied, setCopied] = useState(false);

  const handleCopy = () => {
    navigator.clipboard.writeText(code);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  return (
    <div className="code-block-wrapper">
      <div className="code-block-header">
        <span className="code-lang">{language || "code"}</span>
        <button
          type="button"
          className={`copy-btn ${copied ? "copied" : ""}`}
          onClick={handleCopy}
          title="Copy code"
        >
          {copied ? (
            <>
              <Check size={14} />
              <span>Copied!</span>
            </>
          ) : (
            <>
              <Copy size={14} />
              <span>Copy code</span>
            </>
          )}
        </button>
      </div>
      <pre className="code-block-body">
        <code>{code}</code>
      </pre>
    </div>
  );
}

function renderInlineMarkdown(text) {
  if (!text) return null;

  // Render markdown links [text](url)
  const linkRegex = /\[([^\]]+)\]\(([^)]+)\)/g;
  const parts = [];
  let lastIndex = 0;
  let match;

  while ((match = linkRegex.exec(text)) !== null) {
    if (match.index > lastIndex) {
      parts.push(text.substring(lastIndex, match.index));
    }
    const [_, linkText, linkUrl] = match;
    parts.push(
      <a
        key={match.index}
        href={linkUrl}
        target="_blank"
        rel="noopener noreferrer"
        className="markdown-link"
      >
        {linkText}
      </a>
    );
    lastIndex = linkRegex.lastIndex;
  }
  if (lastIndex < text.length) {
    parts.push(text.substring(lastIndex));
  }

  // Parse inline bold **bold** and inline code `code`
  return parts.map((part, pIdx) => {
    if (typeof part !== "string") return part;

    // split inline code
    const codeParts = part.split(/(`[^`]+`)/g);
    return codeParts.map((sub, sIdx) => {
      if (sub.startsWith("`") && sub.endsWith("`")) {
        return (
          <code key={`${pIdx}-${sIdx}`} className="inline-code">
            {sub.slice(1, -1)}
          </code>
        );
      }

      // split inline bold
      const boldParts = sub.split(/(\*\*[^*]+\*\*)/g);
      return boldParts.map((bSub, bIdx) => {
        if (bSub.startsWith("**") && bSub.endsWith("**")) {
          return <strong key={`${pIdx}-${sIdx}-${bIdx}`}>{bSub.slice(2, -2)}</strong>;
        }
        return bSub;
      });
    });
  });
}

export default function MarkdownMessage({ content }) {
  if (!content) return null;

  // Split by code blocks ```...```
  const regex = /```(\w+)?\n([\s\S]*?)```/g;
  const blocks = [];
  let lastIndex = 0;
  let match;

  while ((match = regex.exec(content)) !== null) {
    if (match.index > lastIndex) {
      blocks.push({
        type: "text",
        value: content.substring(lastIndex, match.index),
      });
    }
    blocks.push({
      type: "code",
      language: match.group ? match.group(1) : match[1] || "",
      code: match[2].trimEnd(),
    });
    lastIndex = regex.lastIndex;
  }

  if (lastIndex < content.length) {
    blocks.push({
      type: "text",
      value: content.substring(lastIndex),
    });
  }

  return (
    <div className="markdown-content">
      {blocks.map((block, index) => {
        if (block.type === "code") {
          return (
            <CodeBlock
              key={index}
              language={block.language}
              code={block.code}
            />
          );
        }

        // Process paragraphs, headers, list items in text block
        const lines = block.value.split("\n");
        return (
          <div key={index} className="text-block">
            {lines.map((line, lIdx) => {
              const trimmed = line.trim();
              if (!trimmed) return <div key={lIdx} className="paragraph-spacer" />;

              if (trimmed.startsWith("# ")) {
                return (
                  <h1 key={lIdx} className="md-h1">
                    {renderInlineMarkdown(trimmed.slice(2))}
                  </h1>
                );
              }
              if (trimmed.startsWith("## ")) {
                return (
                  <h2 key={lIdx} className="md-h2">
                    {renderInlineMarkdown(trimmed.slice(3))}
                  </h2>
                );
              }
              if (trimmed.startsWith("### ")) {
                return (
                  <h3 key={lIdx} className="md-h3">
                    {renderInlineMarkdown(trimmed.slice(4))}
                  </h3>
                );
              }

              if (trimmed.startsWith("- ") || trimmed.startsWith("* ")) {
                return (
                  <li key={lIdx} className="md-bullet">
                    {renderInlineMarkdown(trimmed.slice(2))}
                  </li>
                );
              }

              const numMatch = trimmed.match(/^(\d+)\.\s+(.*)/);
              if (numMatch) {
                return (
                  <div key={lIdx} className="md-numbered">
                    <span className="num-prefix">{numMatch[1]}.</span>
                    <span>{renderInlineMarkdown(numMatch[2])}</span>
                  </div>
                );
              }

              return (
                <p key={lIdx} className="md-p">
                  {renderInlineMarkdown(line)}
                </p>
              );
            })}
          </div>
        );
      })}
    </div>
  );
}
