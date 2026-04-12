/**
 * Browser-side WordPiece tokenizer for BERT-family models.
 * Loads vocab.txt and implements BertNormalizer + BertPreTokenizer + WordPiece.
 */

const CLS_TOKEN = '[CLS]';
const SEP_TOKEN = '[SEP]';
const UNK_TOKEN = '[UNK]';
const PAD_TOKEN = '[PAD]';

export interface TokenizedText {
  inputIds: Int32Array;
  attentionMask: Int32Array;
  /** The actual sequence length before padding */
  seqLength: number;
}

export class WordPieceTokenizer {
  private vocab: Map<string, number> = new Map();
  private loaded = false;

  constructor(private vocabUrl: string) {}

  async load(): Promise<void> {
    if (this.loaded) return;
    const resp = await fetch(this.vocabUrl);
    const text = await resp.text();
    const lines = text.split('\n');
    for (let i = 0; i < lines.length; i++) {
      const token = lines[i]!;
      if (token.length > 0) {
        this.vocab.set(token, i);
      }
    }
    this.loaded = true;
  }

  private tokenToId(token: string): number {
    return this.vocab.get(token) ?? this.vocab.get(UNK_TOKEN) ?? 100;
  }

  /**
   * Tokenize text and return padded input_ids + attention_mask.
   * @param text Input sentence
   * @param maxLength Max sequence length (default 128)
   */
  tokenize(text: string, maxLength = 128): TokenizedText {
    // BertNormalizer: lowercase + clean text
    const normalized = this.normalize(text);

    // BertPreTokenizer: split on whitespace and punctuation
    const words = this.preTokenize(normalized);

    // WordPiece tokenization
    const tokens: string[] = [CLS_TOKEN];
    for (const word of words) {
      const subTokens = this.wordPieceTokenize(word);
      tokens.push(...subTokens);
    }
    tokens.push(SEP_TOKEN);

    // Truncate to maxLength
    const truncated = tokens.slice(0, maxLength);
    const seqLength = truncated.length;

    // Convert to IDs and pad
    const inputIds = new Int32Array(maxLength);
    const attentionMask = new Int32Array(maxLength);
    const padId = this.tokenToId(PAD_TOKEN);

    for (let i = 0; i < maxLength; i++) {
      if (i < truncated.length) {
        inputIds[i] = this.tokenToId(truncated[i]!);
        attentionMask[i] = 1;
      } else {
        inputIds[i] = padId;
        attentionMask[i] = 0;
      }
    }

    return { inputIds, attentionMask, seqLength };
  }

  /** BertNormalizer: lowercase, strip accents, clean control chars */
  private normalize(text: string): string {
    // Lowercase
    let result = text.toLowerCase();
    // Remove control characters (except whitespace)
    result = result.replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, '');
    // Normalize whitespace
    result = result.replace(/\s+/g, ' ').trim();
    return result;
  }

  /** BertPreTokenizer: split on whitespace and punctuation */
  private preTokenize(text: string): string[] {
    const tokens: string[] = [];
    let current = '';

    for (const char of text) {
      if (this.isWhitespace(char)) {
        if (current) {
          tokens.push(current);
          current = '';
        }
      } else if (this.isPunctuation(char)) {
        if (current) {
          tokens.push(current);
          current = '';
        }
        tokens.push(char);
      } else {
        current += char;
      }
    }
    if (current) tokens.push(current);

    return tokens;
  }

  /** WordPiece: greedily match longest subword from vocab */
  private wordPieceTokenize(word: string): string[] {
    if (this.vocab.has(word)) return [word];

    const tokens: string[] = [];
    let start = 0;

    while (start < word.length) {
      let end = word.length;
      let found = false;

      while (start < end) {
        const substr = start === 0
          ? word.slice(start, end)
          : '##' + word.slice(start, end);

        if (this.vocab.has(substr)) {
          tokens.push(substr);
          found = true;
          break;
        }
        end--;
      }

      if (!found) {
        tokens.push(UNK_TOKEN);
        break;
      }
      start = end;
    }

    return tokens;
  }

  private isWhitespace(char: string): boolean {
    return char === ' ' || char === '\t' || char === '\n' || char === '\r';
  }

  private isPunctuation(char: string): boolean {
    const code = char.charCodeAt(0);
    // ASCII punctuation ranges
    return (code >= 33 && code <= 47) ||   // ! " # $ % & ' ( ) * + , - . /
           (code >= 58 && code <= 64) ||   // : ; < = > ? @
           (code >= 91 && code <= 96) ||   // [ \ ] ^ _ `
           (code >= 123 && code <= 126);   // { | } ~
  }
}

/** Shared tokenizer instance — loaded once, reused across benchmarks */
let sharedTokenizer: WordPieceTokenizer | null = null;

export async function getTokenizer(): Promise<WordPieceTokenizer> {
  if (!sharedTokenizer) {
    sharedTokenizer = new WordPieceTokenizer(
      '/distilbert-base-uncased-finetuned-sst-2-english/onnx/vocab.txt'
    );
    await sharedTokenizer.load();
  }
  return sharedTokenizer;
}
