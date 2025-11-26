import os
import base64
import json
from typing import List
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from openai import OpenAI
from datetime import datetime
import re
import fitz  # PyMuPDF


client = OpenAI()  # o client = OpenAI(api_key="...")

app = FastAPI()
templates = Jinja2Templates(directory="templates")


async def extract_invoice_data(image_bytes: bytes) -> dict:
    b64_image = base64.b64encode(image_bytes).decode("utf-8")
    image_url = f"data:image/jpeg;base64,{b64_image}"

    system_prompt = """
Eres un asistente contable especializado en facturación argentina.
Extraes datos de comprobantes (facturas, notas de crédito/débito, tickets, etc.) a partir de una imagen.

Devuelve SIEMPRE un JSON válido con la siguiente estructura (si algún dato no se ve, deja null o "" pero no inventes):

{
  "datos_comprobante": {
    "tipo": "",
    "letra": "",
    "punto_venta": "",
    "numero_comprobante": "",
    "fecha_emision": "",
    "fecha_vencimiento": "",
    "condicion_venta": "",
    "moneda": "",
    "cotizacion_moneda": null
  },
  "emisor": {
    "razon_social": "",
    "cuit": "",
    "domicilio_comercial": "",
    "condicion_iva": "",
    "condicion_ingresos_brutos": "",
    "localidad": "",
    "provincia": "",
    "pais": ""
  },
  "receptor": {
    "razon_social": "",
    "cuit": "",
    "domicilio_comercial": "",
    "condicion_iva": "",
    "condicion_ingresos_brutos": "",
    "tipo_documento": "",
    "numero_documento": ""
  },
  "totales": {
    "importe_neto_gravado": null,
    "importe_neto_no_gravado": null,
    "importe_exento": null,
    "ivAs": [
      {
        "alicuota": 21.0,
        "importe_iva": null
      }
    ],
    "percepciones_iva": null,
    "percepciones_ingresos_brutos": null,
    "percepciones_otras": null,
    "descuentos_generales": null,
    "subtotal": null,
    "total_comprobante": null
  },
  "items": [
    {
      "codigo": "",
      "descripcion": "",
      "unidad_medida": "",
      "cantidad": null,
      "precio_unitario": null,
      "bonificacion": null,
      "alicuota_iva": null,
      "importe_total_renglon": null
    }
  ],
  "datos_fiscales_afip": {
    "cae": "",
    "fecha_vencimiento_cae": "",
    "codigo_barras_qr": "",
    "tipo_documento_receptor": "",
    "numero_documento_receptor": ""
  },
  "datos_compras_importaciones": {
    "condicion_bienes": "",
    "centro_costo": "",
    "numero_remito": "",
    "numero_despacho_importacion": "",
    "gastos_relacionados": ""
  }
}
    """.strip()

    user_prompt = """
Extrae los datos del comprobante de la imagen adjunta.
No expliques nada, solo responde con el JSON pedido.
    """.strip()

    response = client.chat.completions.create(
        model="gpt-4.1-mini",
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": image_url}},
                ],
            },
        ],
    )

async def extract_invoice_data_from_pdf(pdf_bytes: bytes) -> dict:
    """
    Toma un PDF en bytes, renderiza la primera página a imagen
    y reutiliza extract_invoice_data para que la IA lea la factura.
    """
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    if doc.page_count == 0:
        return {"error": "PDF sin páginas"}

    # Por ahora solo usamos la primera página (MVP)
    page = doc.load_page(0)
    pix = page.get_pixmap(dpi=200)  # resolución razonable para OCR
    img_bytes = pix.tobytes("jpeg")

    data = await extract_invoice_data(img_bytes)
    return data

    content = response.choices[0].message.content
    try:
        data = json.loads(content)
    except json.JSONDecodeError:
        data = {"error": "No se pudo parsear la respuesta de la IA", "raw": content}

    return data


# ---------- NUEVO: construcción del .txt para importación ----------

def build_txt_line(data: dict) -> str:
    """Arma una línea de texto con campos separados por ';' para importación masiva."""
    dc = data.get("datos_comprobante", {})
    em = data.get("emisor", {})
    rec = data.get("receptor", {})
    tot = data.get("totales", {})

    # Cuit o DNI del receptor (si es CF con DNI lo podés ajustar en el prompt)
    cuit_o_doc_receptor = rec.get("cuit") or rec.get("numero_documento") or ""

    # Toma el primer IVA como referencia (generalmente 21%)
    ivas = tot.get("ivAs") or []
    iva_21 = ""
    if ivas:
        iva_21 = ivas[0].get("importe_iva") or ""
    
    def to_str(v):
        return "" if v is None else str(v)

    fields = [
        dc.get("fecha_emision", ""),
        dc.get("tipo", ""),
        dc.get("letra", ""),
        dc.get("punto_venta", ""),
        dc.get("numero_comprobante", ""),
        em.get("cuit", ""),
        em.get("razon_social", ""),
        cuit_o_doc_receptor,
        rec.get("razon_social", ""),
        to_str(tot.get("importe_neto_gravado")),
        to_str(iva_21),
        to_str(tot.get("total_comprobante")),
    ]

    # Reemplaza ';' dentro de textos por ',' para no romper el formato
    safe_fields = [str(f).replace(";", ",") for f in fields]

    return ";".join(safe_fields)


def build_txt_content(results: List[dict]) -> str:
    """Arma todo el contenido del .txt a partir de la lista de resultados."""
    lines = []

    # Encabezado (opcional, borralo si tu sistema no lo admite)
    header = (
        "FECHA_EMISION;TIPO;LETRA;PTO_VTA;NRO;"
        "CUIT_EMISOR;RAZON_EMISOR;CUIT_O_DNI_RECEPTOR;"
        "RAZON_RECEPTOR;NETO_GRAVADO;IVA_21;TOTAL"
    )
    lines.append(header)

    for item in results:
        data = item.get("data", {})
        line = build_txt_line(data)
        lines.append(line)

    return "\n".join(lines)

def _to_float(v):
    if v in (None, "", "null"):
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def check_math(data: dict, tol: float = 0.10) -> dict:
    """
    Control matemático básico por comprobante.
    Compara:
      - Neto / IVA / Total calculados desde los ítems
      - contra los totales del JSON devuelto por la IA.
    tol = tolerancia en pesos (por defecto 0,10).
    """
    tot = data.get("totales", {}) or {}
    items = data.get("items", []) or []

    # Totales del JSON
    neto_json = _to_float(tot.get("importe_neto_gravado"))
    exento = _to_float(tot.get("importe_exento"))
    no_grav = _to_float(tot.get("importe_neto_no_gravado"))
    total_json = _to_float(tot.get("total_comprobante"))
    percep_iva = _to_float(tot.get("percepciones_iva"))
    percep_iibb = _to_float(tot.get("percepciones_ingresos_brutos"))
    percep_otras = _to_float(tot.get("percepciones_otras"))

    iva_list = tot.get("ivAs") or []
    iva_json = 0.0
    for iva in iva_list:
        iva_json += _to_float(iva.get("importe_iva"))

    # Recalcular desde ítems (misma lógica que usamos en CItems)
    neto_items = 0.0
    iva_items = 0.0
    total_items = 0.0

    for it in items:
        total_r = _to_float(it.get("importe_total_renglon"))
        alic = it.get("alicuota_iva")
        try:
            alic_f = float(alic) if alic not in (None, "", "null") else 0.0
        except Exception:
            alic_f = 0.0

        if alic_f > 0:
            neto_i = round(total_r / (1 + alic_f / 100), 2)
            iva_i = round(total_r - neto_i, 2)
        else:
            neto_i = total_r
            iva_i = 0.0

        neto_items += neto_i
        iva_items += iva_i
        total_items += total_r

    # Total teórico según totales
    total_teorico = neto_json + iva_json + exento + no_grav + percep_iva + percep_iibb + percep_otras

    # Diferencias
    neto_diff = neto_items - neto_json
    iva_diff = iva_items - iva_json
    total_diff_items_vs_json = total_items - total_json
    total_diff_teorico_vs_json = total_teorico - total_json

    ok = (
        abs(neto_diff) <= tol
        and abs(iva_diff) <= tol
        and abs(total_diff_items_vs_json) <= tol
        and abs(total_diff_teorico_vs_json) <= tol
    )

    return {
        "ok": ok,
        "neto_items": round(neto_items, 2),
        "neto_json": round(neto_json, 2),
        "neto_diff": round(neto_diff, 2),
        "iva_items": round(iva_items, 2),
        "iva_json": round(iva_json, 2),
        "iva_diff": round(iva_diff, 2),
        "total_items": round(total_items, 2),
        "total_json": round(total_json, 2),
        "total_teorico": round(total_teorico, 2),
        "total_diff_items_vs_json": round(total_diff_items_vs_json, 2),
        "total_diff_teorico_vs_json": round(total_diff_teorico_vs_json, 2),
    }

# ---------- Layouts por sistema contable ----------

# ---------- Layout HOLISTOR ----------

# ---------- Layout HOLISTOR (Libro IVA Compras, Anexo I) ----------

def build_txt_line_holistor(data: dict) -> str:
    dc = data.get("datos_comprobante", {})
    em = data.get("emisor", {})
    rec = data.get("receptor", {})
    tot = data.get("totales", {})
    afip = data.get("datos_fiscales_afip", {})

    def s(v):
        return "" if v is None else str(v)

    # Datos básicos
    fecha_emision = dc.get("fecha_emision", "")              # Fecha Emisión
    fecha_recepcion = fecha_emision                          # por ahora usamos la misma

    nombre_comprobante = dc.get("tipo", "")                  # "Factura", "Nota de crédito", etc.
    tipo_comprobante = dc.get("letra", "")                   # A / B / C

    sucursal = dc.get("punto_venta", "")                     # Número Sucursal
    numero = dc.get("numero_comprobante", "")                # Número de Comprobante

    # Totales
    neto_gravado = tot.get("importe_neto_gravado")
    no_gravado = tot.get("importe_neto_no_gravado")
    exento = tot.get("importe_exento")
    total = tot.get("total_comprobante")

    # IVA (sumamos todos los IVA)
    ivas = tot.get("ivAs") or []
    tasa_iva = ""
    iva_liquidado = 0.0
    for iva in ivas:
        if tasa_iva == "" and iva.get("alicuota") is not None:
            tasa_iva = s(iva.get("alicuota"))
        imp = iva.get("importe_iva")
        if isinstance(imp, (int, float)):
            iva_liquidado += imp or 0.0
        elif isinstance(imp, str):
            try:
                iva_liquidado += float(imp.replace(",", "."))
            except Exception:
                pass

    credito_fiscal = iva_liquidado  # en compras suele coincidir

    # Percepciones (sumamos todas las que tengas)
    percep_total = 0.0
    for key in ("percepciones_iva", "percepciones_ingresos_brutos", "percepciones_otras"):
        val = tot.get(key)
        if isinstance(val, (int, float)):
            percep_total += val or 0.0
        elif isinstance(val, str):
            try:
                percep_total += float(val.replace(",", "."))
            except Exception:
                pass

    # Proveedor
    condicion_fiscal = em.get("condicion_iva", "")
    cuit_proveedor = em.get("cuit", "")
    nombre_proveedor = em.get("razon_social", "")
    domicilio_proveedor = em.get("domicilio_comercial", "")
    codigo_postal = ""                                        # no lo tenemos desglosado
    provincia = em.get("provincia", "")

    # Documento
    tipo_doc_cliente = "80" if cuit_proveedor else ""

    # Moneda
    moneda = dc.get("moneda", "") or ""
    tipo_cambio = dc.get("cotizacion_moneda", None)

    # CAI / CAE (Holistor lo llama C.A.I., lo llenamos con el CAE si existe)
    cai = afip.get("cae", "")

    # Códigos (los dejamos vacíos; Holistor puede usar Códigos Auxiliares por Defecto)
    cod_neto_gravado = ""
    cod_conc_no_gravado = ""
    cod_operacion_exenta = ""
    cod_perc_ret_pc = ""

    # Armamos los 28 campos EN ORDEN EXACTO DEL ANEXO I
    fields = [
        nombre_comprobante,         # 1  Nombre Comprobante
        tipo_comprobante,           # 2  Tipo Comprobante
        sucursal,                   # 3  Número Sucursal
        numero,                     # 4  Número de Comprobante
        fecha_emision,              # 5  Fecha Emisión
        fecha_recepcion,            # 6  Fecha Recepción
        cod_neto_gravado,           # 7  Código Neto Gravado
        s(neto_gravado),            # 8  Neto Gravado
        cod_conc_no_gravado,        # 9  Cód. Concepto no Gravado
        s(no_gravado),              # 10 Conceptos no Gravados
        cod_operacion_exenta,       # 11 Cód. Operación Exenta
        s(exento),                  # 12 Operaciones Exentas
        cod_perc_ret_pc,            # 13 Código Perc./Ret./P.Cta
        s(percep_total),            # 14 Percepciones
        s(tasa_iva),                # 15 Tasa I.V.A.
        s(iva_liquidado),           # 16 I.V.A. Liquidado
        s(credito_fiscal),          # 17 Crédito Fiscal
        s(total),                   # 18 Total
        condicion_fiscal,           # 19 Condición Fiscal Proveedor
        cuit_proveedor,             # 20 C.U.I.T. Proveedor
        nombre_proveedor,           # 21 Nombre Proveedor
        domicilio_proveedor,        # 22 Domicilio Proveedor
        codigo_postal,              # 23 Código Postal
        provincia,                  # 24 Provincia
        tipo_doc_cliente,           # 25 Tipo Documento Cliente
        moneda,                     # 26 Moneda
        s(tipo_cambio),             # 27 Tipo Cambio
        cai,                        # 28 C.A.I.
    ]

    safe_fields = [str(f).replace(";", ",") for f in fields]
    return ";".join(safe_fields)


# =================== HOLISTOR ===================

# =================== HOLISTOR ===================

def _num(v):
    """Convierte a string numérico con punto, vacío si no hay dato."""
    if v in (None, "", "null"):
        return ""
    try:
        return f"{float(v):.2f}"
    except Exception:
        return ""


def build_txt_line_holistor(data: dict) -> str:
    """
    Arma UNA línea del TXT de Holistor con este encabezado:

    Nombre Comprobante;Tipo Comprobante;Numero Sucursal;Numero de Comprobante;
    Fecha Emision;Fecha Recepcion;Codigo Neto Gravado;Neto Gravado;
    Cod Concepto no Gravado;Conceptos no Gravados;
    Cod Operacion Exenta;Operaciones Exentas;
    Codigo Perc_Ret_PCta;Percepciones;
    Tasa IVA;IVA Liquidado;Credito Fiscal;Total;
    Condicion Fiscal Proveedor;CUIT Proveedor;Nombre Proveedor;Domicilio Proveedor;
    Codigo Postal;Provincia;Tipo Documento Cliente;Moneda;Tipo Cambio;CAI
    """

    dc = data.get("datos_comprobante", {}) or {}
    em = data.get("emisor", {}) or {}
    rec = data.get("receptor", {}) or {}
    tot = data.get("totales", {}) or {}
    fisc = data.get("datos_fiscales_afip", {}) or {}

    # -------- Totales crudos del JSON --------
    importe_neto_gravado_raw = tot.get("importe_neto_gravado") or 0.0
    importe_no_gravado = tot.get("importe_neto_no_gravado") or 0.0
    importe_exento = tot.get("importe_exento") or 0.0
    total_comprobante = tot.get("total_comprobante") or 0.0

    ivas = tot.get("ivAs") or []
    if ivas:
        iva_row = ivas[0]
        tasa_iva = iva_row.get("alicuota") or 0.0
        iva_liquidado = iva_row.get("importe_iva") or 0.0
    else:
        tasa_iva = 0.0
        iva_liquidado = 0.0

    # Percepciones: sumo todo lo que haya
    perc_iva = tot.get("percepciones_iva") or 0.0
    perc_iibb = tot.get("percepciones_ingresos_brutos") or 0.0
    perc_otras = tot.get("percepciones_otras") or 0.0
    percepciones_total = (perc_iva or 0) + (perc_iibb or 0) + (perc_otras or 0)

    # -------- RE-CÁLCULO MATEMÁTICO DEL NETO GRAVADO --------
    # Fórmula: total = neto_gravado + no_gravado + exento + percepciones + IVA
    # => neto_gravado = (total - no_gravado - exento - percepciones) / (1 + tasa_iva/100)
    try:
        base_total = float(total_comprobante) - float(importe_no_gravado) - float(importe_exento) - float(percepciones_total)
        if float(tasa_iva) > 0:
            neto_calc = round(base_total / (1.0 + float(tasa_iva) / 100.0), 2)
        else:
            # Si no hay IVA, el neto gravado es directamente la base
            neto_calc = round(base_total, 2)
    except Exception:
        # Si algo falla, usamos lo que vino del JSON
        neto_calc = importe_neto_gravado_raw

    importe_neto_gravado = neto_calc
    credito_fiscal = iva_liquidado  # por ahora igual al IVA

    # -------- Campos específicos Holistor --------

    # Nombre / tipo comprobante
    nombre_comprobante = (dc.get("tipo") or "Factura").capitalize()
    tipo_comprobante = dc.get("letra") or ""

    nro_sucursal = (dc.get("punto_venta") or "").zfill(4)
    nro_comprobante = dc.get("numero_comprobante") or ""

    fecha_emision = dc.get("fecha_emision") or ""
    fecha_recepcion = fecha_emision  # por ahora uso misma fecha

    # Códigos para neto / exento / no gravado
    codigo_neto_gravado = "1" if importe_neto_gravado else ""
    cod_concepto_no_gravado = "2" if importe_no_gravado else ""
    cod_operacion_exenta = "3" if importe_exento else ""

    # Código percepción / retención (si hay algo, pongo 1)
    codigo_perc_ret_pcta = "1" if percepciones_total else ""

    condicion_fiscal_prov = em.get("condicion_iva") or ""
    cuit_prov = em.get("cuit") or ""
    nombre_prov = em.get("razon_social") or ""
    domicilio_prov = em.get("domicilio_comercial") or ""
    cp = ""  # No lo tenemos en el JSON todavía
    provincia = em.get("provincia") or ""

    # Tipo documento cliente (80 = CUIT, 96 = DNI, etc.)
    if rec.get("cuit"):
        tipo_doc_cliente = "80"
    elif rec.get("numero_documento"):
        tipo_doc_cliente = "96"
    else:
        tipo_doc_cliente = ""

    moneda = dc.get("moneda") or ""
    tipo_cambio = dc.get("cotizacion_moneda") or ""
    cai_cae = fisc.get("cae") or ""

    campos = [
        nombre_comprobante,                 # Nombre Comprobante
        tipo_comprobante,                   # Tipo Comprobante (letra)
        nro_sucursal,                       # Numero Sucursal
        nro_comprobante,                    # Numero de Comprobante
        fecha_emision,                      # Fecha Emision
        fecha_recepcion,                    # Fecha Recepcion
        codigo_neto_gravado,                # Codigo Neto Gravado
        _num(importe_neto_gravado),         # Neto Gravado (RECALCULADO)
        cod_concepto_no_gravado,           # Cod Concepto no Gravado
        _num(importe_no_gravado),           # Conceptos no Gravados
        cod_operacion_exenta,              # Cod Operacion Exenta
        _num(importe_exento),              # Operaciones Exentas
        codigo_perc_ret_pcta,              # Codigo Perc_Ret_PCta
        _num(percepciones_total),          # Percepciones
        _num(tasa_iva),                    # Tasa IVA
        _num(iva_liquidado),               # IVA Liquidado
        _num(credito_fiscal),              # Credito Fiscal
        _num(total_comprobante),           # Total
        condicion_fiscal_prov,             # Condicion Fiscal Proveedor
        cuit_prov,                         # CUIT Proveedor
        nombre_prov,                       # Nombre Proveedor
        domicilio_prov,                    # Domicilio Proveedor
        cp,                                # Codigo Postal
        provincia,                         # Provincia
        tipo_doc_cliente,                  # Tipo Documento Cliente
        moneda,                            # Moneda
        _num(tipo_cambio),                 # Tipo Cambio
        cai_cae,                           # CAI / CAE
    ]

    # reemplazo ';' internos por ',' para no romper el CSV
    safe_campos = [str(c).replace(";", ",") for c in campos]
    return ";".join(safe_campos)


def build_txt_content_holistor(results: List[dict]) -> str:
    """Arma TODO el TXT de Holistor (con encabezado + líneas)."""
    header = (
        "Nombre Comprobante;Tipo Comprobante;Numero Sucursal;Numero de Comprobante;"
        "Fecha Emision;Fecha Recepcion;Codigo Neto Gravado;Neto Gravado;"
        "Cod Concepto no Gravado;Conceptos no Gravados;"
        "Cod Operacion Exenta;Operaciones Exentas;"
        "Codigo Perc_Ret_PCta;Percepciones;"
        "Tasa IVA;IVA Liquidado;Credito Fiscal;Total;"
        "Condicion Fiscal Proveedor;CUIT Proveedor;Nombre Proveedor;Domicilio Proveedor;"
        "Codigo Postal;Provincia;Tipo Documento Cliente;Moneda;Tipo Cambio;CAI"
    )

    lines = [header]

    for item in results:
        data = item.get("data", {}) or {}
        line = build_txt_line_holistor(data)
        lines.append(line)

    return "\n".join(lines)


# =================== FIN HOLISTOR ===================


# =================== BEJERMAN ===================
def _to_float(v):
    if v in (None, "", "null"):
        return 0.0
    try:
        return float(v)
    except Exception:
        return 0.0


def check_math(data: dict, tol: float = 0.10) -> dict:
    """
    Control matemático básico por comprobante.
    Compara:
      - Neto / IVA / Total calculados desde los ítems
      - contra los totales del JSON devuelto por la IA.
    tol = tolerancia en pesos (por defecto 0,10).
    """
    tot = data.get("totales", {}) or {}
    items = data.get("items", []) or []

    # Totales del JSON
    neto_json = _to_float(tot.get("importe_neto_gravado"))
    exento = _to_float(tot.get("importe_exento"))
    no_grav = _to_float(tot.get("importe_neto_no_gravado"))
    total_json = _to_float(tot.get("total_comprobante"))
    percep_iva = _to_float(tot.get("percepciones_iva"))
    percep_iibb = _to_float(tot.get("percepciones_ingresos_brutos"))
    percep_otras = _to_float(tot.get("percepciones_otras"))

    iva_list = tot.get("ivAs") or []
    iva_json = 0.0
    for iva in iva_list:
        iva_json += _to_float(iva.get("importe_iva"))

    # Recalcular desde ítems
    neto_items = 0.0
    iva_items = 0.0
    total_items = 0.0

    for it in items:
        total_r = _to_float(it.get("importe_total_renglon"))
        alic = it.get("alicuota_iva")
        try:
            alic_f = float(alic) if alic not in (None, "", "null") else 0.0
        except Exception:
            alic_f = 0.0

        if alic_f > 0:
            neto_i = round(total_r / (1 + alic_f / 100), 2)
            iva_i = round(total_r - neto_i, 2)
        else:
            neto_i = total_r
            iva_i = 0.0

        neto_items += neto_i
        iva_items += iva_i
        total_items += total_r

    total_teorico = (
        neto_json + iva_json + exento + no_grav +
        percep_iva + percep_iibb + percep_otras
    )

    neto_diff = neto_items - neto_json
    iva_diff = iva_items - iva_json
    total_diff_items_vs_json = total_items - total_json
    total_diff_teorico_vs_json = total_teorico - total_json

    ok = (
        abs(neto_diff) <= tol
        and abs(iva_diff) <= tol
        and abs(total_diff_items_vs_json) <= tol
        and abs(total_diff_teorico_vs_json) <= tol
    )

    return {
        "ok": ok,
        "neto_items": round(neto_items, 2),
        "neto_json": round(neto_json, 2),
        "neto_diff": round(neto_diff, 2),
        "iva_items": round(iva_items, 2),
        "iva_json": round(iva_json, 2),
        "iva_diff": round(iva_diff, 2),
        "total_items": round(total_items, 2),
        "total_json": round(total_json, 2),
        "total_teorico": round(total_teorico, 2),
        "total_diff_items_vs_json": round(total_diff_items_vs_json, 2),
        "total_diff_teorico_vs_json": round(total_diff_teorico_vs_json, 2),
    }

# =================== BEJERMAN: HELPERS GENERALES ===================

def _pad_right(value: str, length: int, fill: str = " ") -> str:
    """Rellena a la derecha con blancos."""
    s = "" if value is None else str(value)
    return s[:length].ljust(length, fill)


def _pad_left(value: str, length: int, fill: str = "0") -> str:
    """Rellena a la izquierda (típico para números)."""
    s = "" if value is None else str(value)
    return s[:length].rjust(length, fill)


def _only_digits(value: str) -> str:
    if not value:
        return ""
    return "".join(ch for ch in str(value) if ch.isdigit())


def _format_date_yyyymmdd(date_str: str) -> str:
    """Convierte '03/07/2025', '2025-07-03', etc. a '20250703'. Si falla, 00000000."""
    if not date_str:
        return "00000000"
    date_str = date_str.strip()
    try:
        if "-" in date_str:
            dt = datetime.strptime(date_str[:10], "%Y-%m-%d")
        elif "/" in date_str:
            parts = date_str.split("/")
            if len(parts[0]) == 4:
                dt = datetime.strptime(date_str[:10], "%Y/%m/%d")
            else:
                dt = datetime.strptime(date_str[:10], "%d/%m/%Y")
        else:
            if len(date_str) == 8:
                dt = datetime.strptime(date_str, "%d%m%Y")
            else:
                return "00000000"
        return dt.strftime("%Y%m%d")
    except Exception:
        return "00000000"


def _format_amount_bejerman(value, length: int = 16) -> str:
    """
    Formato 999999999999.99 (16 caracteres).
    Si no hay dato, devuelve todo ceros.
    """
    if value in (None, "", "null"):
        s = "0.00"
    else:
        try:
            num = float(value)
            s = f"{num:.2f}"
        except Exception:
            s = "0.00"
    return s.replace(",", ".")[:length].rjust(length, "0")

def _get_item_aliquota(item: dict, tot: dict):
    """
    Devuelve la alícuota de IVA del ítem.
    Prioridad:
      1) item["alicuota_iva"] si viene en el JSON
      2) Si no, y en 'totales.ivAs' hay UNA sola alícuota, usa esa
      3) Si no hay nada claro, devuelve 0 (se tratará como exento / no gravado)
    """
    alic = item.get("alicuota_iva")
    if alic not in (None, "", "null"):
        return alic

    ivas = tot.get("ivAs") or []
    if len(ivas) == 1:
        return ivas[0].get("alicuota", 0) or 0

    return 0

# Mapas para provincia e IVA según Bejerman
_BEJ_PROV_MAP = {
    "CAPITAL FEDERAL": "001",
    "CABA": "001",
    "CIUDAD AUTONOMA DE BUENOS AIRES": "001",
    "BUENOS AIRES": "002",
    "CATAMARCA": "003",
    "CORDOBA": "004",
    "CORRIENTES": "005",
    "CHACO": "006",
    "CHUBUT": "007",
    "ENTRE RIOS": "008",
    "FORMOSA": "009",
    "JUJUY": "010",
    "LA PAMPA": "011",
    "LA RIOJA": "012",
    "MENDOZA": "013",
    "MISIONES": "014",
    "NEUQUEN": "015",
    "RIO NEGRO": "016",
    "SALTA": "017",
    "SAN JUAN": "018",
    "SAN LUIS": "019",
    "SANTA CRUZ": "020",
    "SANTA FE": "021",
    "SANTIAGO DEL ESTERO": "022",
    "TIERRA DEL FUEGO": "023",
    "TUCUMAN": "024",
    "EXTERIOR": "025",
}


def _map_provincia_bejerman(nombre: str) -> str:
    if not nombre:
        return "000"
    key = nombre.upper().strip()
    return _BEJ_PROV_MAP.get(key, "000")


_BEJ_IVA_MAP = {
    "IVA RESPONSABLE INSCRIPTO": "1",
    "RESPONSABLE INSCRIPTO": "1",
    "RESPONSABLE INSCRIPTO.": "1",
    "RESPONSABLE MONOTRIBUTO": "6",
    "MONOTRIBUTO": "6",
    "MONOTRIBUTO SOCIAL": "6",
    "CONSUMIDOR FINAL": "3",
    "EXENTO": "5",
    "NO RESPONSABLE": "4",
    "SUJETO NO CATEGORIZADO": "7",
}


def _map_iva_bejerman(condicion_iva: str) -> str:
    if not condicion_iva:
        return "1"  # default inscripto
    key = condicion_iva.upper().strip()
    return _BEJ_IVA_MAP.get(key, "1")


def _map_tipo_comprobante_bejerman(tipo: str) -> str:
    """
    Mapea 'FACTURA', 'NOTA DE CREDITO', etc. a códigos del ASCII (FC, NC, ND...).
    """
    if not tipo:
        return "FC"
    t = tipo.upper()
    if "FACTURA" in t:
        return "FC"
    if "NC" in t or "CRÉDITO" in t or "CREDITO" in t:
        return "NC"
    if "ND" in t or "DÉBITO" in t or "DEBITO" in t:
        return "ND"
    if "ORDEN DE PAGO" in t or "OP" in t:
        return "OP"
    return "FC"


# =================== BEJERMAN: CCabecer.txt ===================

def build_bejerman_ccabecer_line(data: dict) -> str:
    """
    Construye UNA línea de CCabecer.txt (cabecera de compras) en formato ancho fijo.
    """
    dc = data.get("datos_comprobante", {}) or {}
    em = data.get("emisor", {}) or {}

    tipo_comp = _map_tipo_comprobante_bejerman(dc.get("tipo", ""))
    letra = (dc.get("letra") or " ").strip()[:1] or " "
    pto_vta = _pad_left(dc.get("punto_venta", ""), 4, "0")
    nro_comp = _pad_left(dc.get("numero_comprobante", ""), 8, "0")
    nro_hasta = _pad_left("", 8, "0")

    fecha_comp = _format_date_yyyymmdd(dc.get("fecha_emision", ""))

    cod_proveedor = _pad_right("@@@#@@", 6)  # recodificación automática
    razon_prov = _pad_right(em.get("razon_social", ""), 40)

    tipo_doc = _pad_left("1", 2, "0")  # 1 = CUIT

    provincia_cod = _pad_left(
        _map_provincia_bejerman(em.get("provincia", "")),
        3,
        "0",
    )

    situacion_iva = _map_iva_bejerman(em.get("condicion_iva", ""))

    cuit_emisor = _only_digits(em.get("cuit", ""))
    cuit_emisor = _pad_left(cuit_emisor, 11, "0")

    nro_iibb = _pad_right(em.get("condicion_ingresos_brutos", ""), 15)

    clasif1 = _pad_right("", 4)
    clasif2 = _pad_right("", 4)

    condicion_pago = _pad_left("1", 3, "0")  # 1=contado (ajustable)
    cod_causa_emision = _pad_right("", 4)

    fecha_vto = _format_date_yyyymmdd(dc.get("fecha_vencimiento", ""))

    tot = data.get("totales", {}) or {}
    importe_total = _format_amount_bejerman(tot.get("total_comprobante"))

    apertura_contable = _pad_right("", 4)

    direccion = _pad_right(em.get("domicilio_comercial", ""), 30)
    cod_postal = _pad_right("", 8)
    localidad = _pad_right(em.get("localidad", ""), 25)

    actualiza_stock = "N"

    desc_clasif1 = _pad_right("", 15)
    desc_clasif2 = _pad_right("", 15)

    tasa_dto_com_1 = _pad_left("0", 8, "0")
    tasa_dto_com_2 = _pad_left("0", 8, "0")
    tasa_dto_com_3 = _pad_left("0", 8, "0")
    tasa_dto_fin = _pad_left("0", 8, "0")

    aduana = _pad_right("", 8)
    fecha_despa_plaza = "00000000"
    anio_doc = _pad_left("", 4, "0")
    nro_despacho = _pad_right("", 25)
    tipo_declar_import = " "

    campos = [
        _pad_right(tipo_comp, 3),
        _pad_right(letra, 1),
        pto_vta,
        nro_comp,
        nro_hasta,
        fecha_comp,
        _pad_right(cod_proveedor, 6),
        razon_prov,
        tipo_doc,
        provincia_cod,
        situacion_iva,
        cuit_emisor,
        nro_iibb,
        clasif1,
        clasif2,
        condicion_pago,
        cod_causa_emision,
        fecha_vto,
        importe_total,
        apertura_contable,
        direccion,
        cod_postal,
        localidad,
        actualiza_stock,
        desc_clasif1,
        desc_clasif2,
        tasa_dto_com_1,
        tasa_dto_com_2,
        tasa_dto_com_3,
        tasa_dto_fin,
        aduana,
        fecha_despa_plaza,
        anio_doc,
        nro_despacho,
        tipo_declar_import,
    ]

    return "".join(campos)


def build_txt_content_bejerman(results: List[dict]) -> str:
    """
    Genera el contenido de CCabecer.txt para Bejerman.
    (Sólo cabecera; ítems/regímenes/medios de pago se pueden agregar después).
    """
    lines = []
    for item in results:
        data = item.get("data", {})
        line = build_bejerman_ccabecer_line(data)
        lines.append(line)
    return "\n".join(lines)


# =================== BEJERMAN: CItems.txt (detalle de compras) ===================

def build_bejerman_citems_line(data: dict, item: dict) -> str:
    """
    Construye UNA línea de CItems.txt (detalle de comprobantes de compras).
    Un registro por renglón de ítem.
    """
    dc = data.get("datos_comprobante", {}) or {}
    em = data.get("emisor", {}) or {}
    tot = data.get("totales", {}) or {}

    tipo_comp = _map_tipo_comprobante_bejerman(dc.get("tipo", ""))
    letra = (dc.get("letra") or " ").strip()[:1] or " "
    pto_vta = _pad_left(dc.get("punto_venta", ""), 4, "0")
    nro_comp = _pad_left(dc.get("numero_comprobante", ""), 8, "0")
    nro_hasta = _pad_left("", 8, "0")
    fecha_comp = _format_date_yyyymmdd(dc.get("fecha_emision", ""))

    cod_proveedor = _pad_right("@@@#@@", 6)

    # Tipo de ítem: "C" = concepto (no mueve stock)
    tipo_item = "C"

    codigo_item = _pad_right(item.get("codigo", "") or "", 23)

    cantidad = item.get("cantidad") or 1
    cantidad_str = _format_amount_bejerman(cantidad, length=16)

    cant_um2 = _format_amount_bejerman(0, length=16)

    descripcion = _pad_right(item.get("descripcion", "") or "", 50)

    precio_unit = item.get("precio_unitario") or item.get("importe_total_renglon") or 0
    precio_unit_str = _format_amount_bejerman(precio_unit, length=16)

    # ---------- NUEVO: decidir alícuota usando item + totales.ivAs ----------
    alic = _get_item_aliquota(item, tot)
    try:
        alic_float = float(alic)
    except Exception:
        alic_float = 0.0

    tasa_iva_insc = _format_amount_bejerman(alic_float, length=8)
    tasa_iva_no_insc = _format_amount_bejerman(0, length=8)

    # Importe del renglón "final" (con IVA si proveedor inscripto)
    importe_renglon_final = item.get("importe_total_renglon")
    if importe_renglon_final is None:
        # fallback: total_comprobante si hay solo un ítem
        importe_renglon_final = tot.get("total_comprobante") or 0
    try:
        importe_renglon_final = float(importe_renglon_final)
    except Exception:
        importe_renglon_final = 0.0

    # ---------- NUEVO: cálculo neto/IVA según tipo ----------
    if alic_float > 0:
        # Gravado: neto = total / (1 + tasa), IVA = diferencia
        neto_item = round(importe_renglon_final / (1 + alic_float / 100), 2)
        iva_item = round(importe_renglon_final - neto_item, 2)
    else:
        # Tasa 0: exento o no gravado
        neto_item = importe_renglon_final
        iva_item = 0.0

    importe_iva_insc = _format_amount_bejerman(iva_item, length=16)
    importe_iva_no_insc = _format_amount_bejerman(0, length=16)

    importe_total_neto = _format_amount_bejerman(neto_item, length=16)

    # Descuentos en cero por ahora
    dto_com = _format_amount_bejerman(0, length=16)
    dto_fin = _format_amount_bejerman(0, length=16)

    # ---------- NUEVO: tipo_iva más expresivo ----------
    # Regla simple:
    #   >0     → 1 (gravado)
    #   ==0 y hay importe_exento en totales → 2 (exento)
    #   ==0 y hay neto_no_gravado           → 3 (no gravado)
    #   resto                               → 2 por defecto
    tipo_iva = "1"
    if alic_float == 0:
        importe_exento = tot.get("importe_exento") or 0
        importe_no_grav = tot.get("importe_neto_no_gravado") or 0
        try:
            importe_exento = float(importe_exento)
        except Exception:
            importe_exento = 0.0
        try:
            importe_no_grav = float(importe_no_grav)
        except Exception:
            importe_no_grav = 0.0

        if importe_exento > 0 and importe_no_grav == 0:
            tipo_iva = "2"   # exento
        elif importe_no_grav > 0 and importe_exento == 0:
            tipo_iva = "3"   # no gravado
        else:
            tipo_iva = "2"   # default 0% = exento

    cod_concepto_no_grav = _pad_left("", 4)
    importe_no_grav = _format_amount_bejerman(0, length=16)

    dto_por_linea = _format_amount_bejerman(0, length=16)

    deposito = _pad_left("", 3)
    partida = _pad_right("", 26)

    tasa_dto_item = _format_amount_bejerman(0, length=8)

    importe_renglon = _format_amount_bejerman(importe_renglon_final, length=16)

    # Imputación crédito fiscal y rubro: defaults razonables
    imputacion_cf = "1"  # 1 = Directo gravado
    rubro_cf = "0"       # 0 = Compra mercado local

    campos = [
        _pad_right(tipo_comp, 3),      # 1 Tipo Comprobante
        _pad_right(letra, 1),          # 2 Letra
        pto_vta,                       # 3 Punto de venta (4)
        nro_comp,                      # 4 Número comprobante (8)
        nro_hasta,                     # 5 Nro hasta (8)
        fecha_comp,                    # 6 Fecha comprobante (8)
        _pad_right(cod_proveedor, 6),  # 7 Código proveedor (6)
        tipo_item,                     # 8 Tipo de ítem (1)
        codigo_item,                   # 9 Código concepto/artículo (23)
        cantidad_str,                  # 10 Cantidad UM1 (16)
        cant_um2,                      # 11 Cantidad UM2 (16)
        descripcion,                   # 12 Descripción (50)
        precio_unit_str,               # 13 Precio unitario (16)
        tasa_iva_insc,                 # 14 Tasa IVA inscripto (8)
        tasa_iva_no_insc,              # 15 Tasa IVA no inscripto (8)
        importe_iva_insc,              # 16 Importe IVA inscripto (16)
        importe_iva_no_insc,           # 17 Importe IVA no inscripto (16)
        importe_total_neto,            # 18 Importe total neto (16)
        dto_com,                       # 19 Importe dto comercial (16)
        dto_fin,                       # 20 Importe dto financiero (16)
        cod_concepto_no_grav,          # 21 Código concepto no gravado (4)
        importe_no_grav,               # 22 Importe no gravado (16)
        tipo_iva,                      # 23 Tipo de IVA (1)
        dto_por_linea,                 # 24 Importe dto por línea (16)
        deposito,                      # 25 Depósito (3)
        partida,                       # 26 Partida (26)
        tasa_dto_item,                 # 27 Tasa dto por ítem (8)
        importe_renglon,               # 28 Importe del renglón (16)
        imputacion_cf,                 # 29 Imputación crédito fiscal (1)
        rubro_cf,                      # 30 Rubro crédito fiscal (1)
    ]

    return "".join(campos)



def build_txt_citems_bejerman(results: List[dict]) -> str:
    """
    Genera el contenido de CItems.txt para Bejerman.
    Un registro por ítem de cada comprobante.
    """
    lines = []
    for entry in results:
        data = entry.get("data", {}) or {}
        items = data.get("items") or []
        if not items:
            # Si no hay ítems, igual generamos un renglón "dummy" para no dejar el comprobante colgado
            items = [{}]

        for it in items:
            line = build_bejerman_citems_line(data, it)
            lines.append(line)

    return "\n".join(lines)

# =================== BEJERMAN: CRegEsp.txt (regímenes especiales) ===================

def build_bejerman_cregesp_line(data: dict, codigo_regimen: str, codigo_articulo: str, importe) -> str:
    """
    UNA línea de CRegEsp.txt (retenciones / percepciones).
    Un registro por régimen especial.
    """
    dc = data.get("datos_comprobante", {}) or {}

    tipo_comp = _map_tipo_comprobante_bejerman(dc.get("tipo", ""))
    letra = (dc.get("letra") or " ").strip()[:1] or " "
    pto_vta = _pad_left(dc.get("punto_venta", ""), 4, "0")
    nro_comp = _pad_left(dc.get("numero_comprobante", ""), 8, "0")
    nro_hasta = _pad_left("", 8, "0")
    fecha_comp = _format_date_yyyymmdd(dc.get("fecha_emision", ""))

    # recodificación automática de proveedor
    cod_proveedor = _pad_right("@@@#@@", 6)

    cod_regimen = _pad_left(codigo_regimen, 4, "0")
    cod_articulo = _pad_left(codigo_articulo, 4, "0")

    importe_str = _format_amount_bejerman(importe, length=16)  # sin signo, según doc

    campos = [
        _pad_right(tipo_comp, 3),   # 1 Tipo de comprobante
        _pad_right(letra, 1),       # 2 Letra
        pto_vta,                    # 3 Punto de venta (4)
        nro_comp,                   # 4 Nro comprobante (8)
        nro_hasta,                  # 5 Nro hasta (8)
        fecha_comp,                 # 6 Fecha comprobante (8)
        _pad_right(cod_proveedor, 6),  # 7 Código proveedor (6)
        cod_regimen,                # 8 Código régimen especial (4)
        cod_articulo,               # 9 Código artículo (4)
        importe_str,                # 10 Importe (16)
    ]

    return "".join(campos)


def build_txt_cregesp_bejerman(results: List[dict]) -> str:
    """
    Genera el contenido de CRegEsp.txt para Bejerman.
    Usa las percepciones del JSON:
      - percepciones_iva
      - percepciones_ingresos_brutos
      - percepciones_otras
    y las mapea a códigos de régimen genéricos 0001 / 0002 / 0003.
    Después, en Bejerman vos das de alta esos códigos.
    """
    lines = []

    for entry in results:
        data = entry.get("data", {}) or {}
        tot = data.get("totales", {}) or {}

        # Ajustá estos códigos a como los cargues en Bejerman
        mapping = [
            ("percepciones_iva", "0001", "0001"),
            ("percepciones_ingresos_brutos", "0002", "0002"),
            ("percepciones_otras", "0003", "0003"),
        ]

        for key, cod_reg, cod_art in mapping:
            importe = tot.get(key)
            try:
                importe = float(importe) if importe not in (None, "", "null") else 0.0
            except Exception:
                importe = 0.0

            if importe != 0:
                line = build_bejerman_cregesp_line(data, cod_reg, cod_art, importe)
                lines.append(line)

    return "\n".join(lines)


@app.post("/upload", response_class=HTMLResponse)
async def upload_invoices(
    request: Request,
    sistema: str = Form(...),
    files: List[UploadFile] = File(...)
):
    results = []

    for file in files:
        content_type = file.content_type or ""
        file_bytes = await file.read()

        # Imagen (jpg, png, etc.)
        if content_type.startswith("image/"):
            data = await extract_invoice_data(file_bytes)

        # PDF
        elif content_type == "application/pdf":
            data = await extract_invoice_data_from_pdf(file_bytes)

        # Otro formato: lo marcamos como no soportado
        else:
            data = {"error": f"Tipo de archivo no soportado: {content_type}"}

        results.append(
            {
                "filename": file.filename,
                "data": data,
            }
        )

    # TXT principal (Holistor / Bejerman / Tango)
    txt_content = ""
    txt_citems_bejerman = ""
    txt_cregesp_bejerman = ""

    if sistema == "holistor":
        txt_content = build_txt_content_holistor(results)

    elif sistema == "bejerman":
        txt_content = build_txt_content_bejerman(results)
        txt_citems_bejerman = build_txt_citems_bejerman(results)
        txt_cregesp_bejerman = build_txt_cregesp_bejerman(results)

    elif sistema == "tango":
        txt_content = build_txt_content_tango(results)

    return templates.TemplateResponse(
        "results.html",
        {
            "request": request,
            "results": results,
            "txt_content": txt_content,
            "txt_citems_bejerman": txt_citems_bejerman,
            "txt_cregesp_bejerman": txt_cregesp_bejerman,
            "sistema": sistema,
        },
    )


# =================== Aca termina BEJERMAN ===================

def build_txt_content_tango(results: List[dict]) -> str:
    """
    Placeholder para Tango.
    Igual: después se adapta al layout real que nos pases.
    """
    return build_txt_content(results)
    

# ----------------- Rutas -----------------

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


#@app.post("/upload", response_class=HTMLResponse)
#async def upload_invoices(
#    request: Request,
#    files: List[UploadFile] = File(...)
#):
#    results = []

#    for file in files:
#        image_bytes = await file.read()
#        data = await extract_invoice_data(image_bytes)
#        results.append(
#            {
#                "filename": file.filename,
#                "data": data,
#            }
#        )

    # <<< NUEVO: generar contenido .txt
#    txt_content = build_txt_content(results)

    #return templates.TemplateResponse(
    #    "results.html",
    #    {
    #        "request": request,
    #        "results": results,
    #        "txt_content": txt_content,  # <<< NUEVO
        #    },
    #)


# 🔴 ACÁ ES DONDE TENÉS QUE CAMBIAR LA FUNCIÓN

